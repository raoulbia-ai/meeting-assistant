import os
import pyaudio
import json
import websockets
import asyncio
from dotenv import load_dotenv
import signal
import sys
import base64
import uuid
import numpy as np

# Load environment variables
load_dotenv()

# Set OpenAI API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio settings
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000  # 32000 Hz as per mic specs

class VoiceAssistant:
    def __init__(self):
        self.full_transcript = ""
        self.audio_buffer = b""
        self.is_speaking = False
        self.frames_above_threshold = 0
        self.frames_below_threshold = 0
        self.silence_threshold = 500
        self.speech_start_frames = 5
        self.speech_end_frames = 20
        self.question = ""

    def handle_exit(self, signal, frame):
        print("\nExiting the application.")
        sys.exit(0)

    def generate_event_id(self):
        return f"event_{uuid.uuid4().hex[:6]}"

    async def initialize_session(self, websocket):
        session_update = {
            "event_id": self.generate_event_id(),
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """You are a helpful assistant.
                                    Act as a Computer Science and Fullsyack Developer.
                                    Answer questions matter of fact, and be concise""",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "temperature": 0.6,
                "language": "en-US"
            }
        }
        await websocket.send(json.dumps(session_update))
        await websocket.recv()  # Wait for session initialization response

    def select_audio_device(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
        device_index = int(input("Enter the device index for your microphone: "))
        p.terminate()
        return device_index

    async def stream_audio_to_api(self, websocket):
        p = pyaudio.PyAudio()
        device_index = self.select_audio_device()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

        print("Listening... (Ctrl+C to exit)")

        try:
            while True:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                volume = np.abs(audio_array).mean()

                if volume > self.silence_threshold:
                    self.frames_above_threshold += 1
                    self.frames_below_threshold = 0
                    if self.frames_above_threshold >= self.speech_start_frames:
                        self.is_speaking = True
                else:
                    self.frames_below_threshold += 1
                    self.frames_above_threshold = 0
                    if self.frames_below_threshold >= self.speech_end_frames:
                        self.is_speaking = False

                if self.is_speaking:
                    self.audio_buffer += audio_data
                elif not self.is_speaking and self.audio_buffer:
                    await self.send_audio_buffer(websocket)

                await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def send_audio_buffer(self, websocket):
        if len(self.audio_buffer) > 0:
            encoded_audio = base64.b64encode(self.audio_buffer).decode("utf-8")
            append_message = {
                "event_id": self.generate_event_id(),
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            await websocket.send(json.dumps(append_message))

            commit_message = {
                "event_id": self.generate_event_id(),
                "type": "input_audio_buffer.commit"
            }
            await websocket.send(json.dumps(commit_message))
            self.audio_buffer = b""

    def process_transcript_delta(self, message):
        transcript_delta = message.get('delta', '')
        self.full_transcript += transcript_delta
        if not self.question:
            self.question = self.full_transcript
        else:
            print(transcript_delta, end='', flush=True)
            # Check if the response has ended
        if transcript_delta.strip().endswith('.') \
            or transcript_delta.strip().endswith('?') \
            or transcript_delta.strip().endswith('!'):
            print('\n')  # Print a new line after the response
            

    async def openai_realtime_api_interaction(self, websocket):
        try:
            while True:
                response = await websocket.recv()
                if isinstance(response, str):
                    message = json.loads(response)
                    if message['type'] == 'response.audio_transcript.delta':
                        self.process_transcript_delta(message)
                        
        except Exception:
            pass

    async def keep_alive(self, websocket):
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(20)
            except:
                break

    async def main(self):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        while True:
            try:
                async with websockets.connect(OPENAI_REALTIME_API_URL, extra_headers=headers, timeout=30) as websocket:
                    await self.initialize_session(websocket)
                    
                    keep_alive_task = asyncio.create_task(self.keep_alive(websocket))
                    send_audio_task = asyncio.create_task(self.stream_audio_to_api(websocket))
                    receive_response_task = asyncio.create_task(self.openai_realtime_api_interaction(websocket))

                    await asyncio.gather(keep_alive_task, send_audio_task, receive_response_task)

            except Exception:
                pass


if __name__ == "__main__":
    assistant = VoiceAssistant()
    signal.signal(signal.SIGINT, assistant.handle_exit)
    asyncio.run(assistant.main())