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
import webrtcvad
import time
import logging

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

CHUNK = 960
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000

log_file = '/var/log/voice_assistant.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class VoiceAssistant:
    def __init__(self):
        self.full_transcript = ""
        self.audio_buffer = b""
        self.speech_buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.question = ""
        self.vad = webrtcvad.Vad(2)
        self.frame_duration_ms = 30
        self.min_speech_duration = 2.0  # Increased to 2 seconds
        self.max_pause_duration = 0.3  # Keep as is
        self.speech_extension_duration = 1.0  # Allow 1 second for speech to continue after min duration
        self.last_api_call = 0
        self.api_call_cooldown = 2.0
        logging.info("VoiceAssistant initialized with updated parameters")

    def handle_exit(self, signal, frame):
        logging.info("Exiting the application.")
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
                                    Act as a Computer Science and Fullstack Developer.
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
        await websocket.recv()
        logging.info("Session initialized")

    def select_audio_device(self):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
        device_index = int(input("Enter the device index for your microphone: "))
        p.terminate()
        logging.info(f"Selected audio device index: {device_index}")
        return device_index

    def is_speech(self, audio_segment):
        return self.vad.is_speech(audio_segment, RATE)

    async def process_audio(self, audio_data, websocket):
        is_speech = self.is_speech(audio_data)
        if is_speech or self.silence_frames * self.frame_duration_ms / 1000 <= self.max_pause_duration:
            self.speech_buffer += audio_data
            self.speech_frames += 1
            self.silence_frames = 0 if is_speech else self.silence_frames + 1
            speech_duration = self.speech_frames * self.frame_duration_ms / 1000
            logging.debug(f"Speech detected, duration: {speech_duration:.2f}s")

            if speech_duration >= self.min_speech_duration:
                # Continue capturing for the extension duration
                await asyncio.sleep(self.speech_extension_duration)
                logging.info(f"Speech duration {speech_duration:.2f}s meets minimum. Sending to API.")
                await self.send_audio_buffer(websocket)
                self.reset_speech_detection()
        elif self.speech_buffer:
            logging.debug(f"Speech ended, duration: {speech_duration:.2f}s. Discarding (below minimum).")
            self.reset_speech_detection()

    def reset_speech_detection(self):
        self.speech_buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0

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
        logging.info("Started listening for audio input")

        try:
            while True:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                await self.process_audio(audio_data, websocket)
                await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logging.info("Stopped listening for audio input")

    async def send_audio_buffer(self, websocket):
        current_time = time.time()
        if current_time - self.last_api_call >= self.api_call_cooldown:
            if len(self.speech_buffer) > 0:
                logging.info("Sending audio buffer to API")
                encoded_audio = base64.b64encode(self.speech_buffer).decode("utf-8")
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
                self.speech_buffer = b""
            self.last_api_call = current_time
        else:
            logging.info("API call skipped due to cooldown")

    def process_transcript_delta(self, message):
        transcript_delta = message.get('delta', '')
        self.full_transcript += transcript_delta
        if not self.question:
            self.question = self.full_transcript
        else:
            print(transcript_delta, end='', flush=True)
        if transcript_delta.strip().endswith('.') \
            or transcript_delta.strip().endswith('?') \
            or transcript_delta.strip().endswith('!'):
            print('\n')
        logging.debug(f"Processed transcript delta: {transcript_delta}")

    async def openai_realtime_api_interaction(self, websocket):
        try:
            while True:
                response = await websocket.recv()
                logging.debug("Received response from API")
                if isinstance(response, str):
                    message = json.loads(response)
                    if message['type'] == 'response.audio_transcript.delta':
                        self.process_transcript_delta(message)
                    logging.debug(f"Processed message type: {message['type']}")
        except Exception as e:
            logging.error(f"Error in API interaction: {str(e)}")

    async def keep_alive(self, websocket):
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(60)
                logging.debug("Sent keep-alive ping")
            except Exception as e:
                logging.error(f"Error in keep-alive: {str(e)}")
                break

    async def main(self):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        while True:
            try:
                logging.info("Attempting to connect to OpenAI API")
                async with websockets.connect(OPENAI_REALTIME_API_URL, extra_headers=headers, timeout=30) as websocket:
                    logging.info("Connected to OpenAI API")
                    await self.initialize_session(websocket)
                    
                    keep_alive_task = asyncio.create_task(self.keep_alive(websocket))
                    send_audio_task = asyncio.create_task(self.stream_audio_to_api(websocket))
                    receive_response_task = asyncio.create_task(self.openai_realtime_api_interaction(websocket))

                    await asyncio.gather(keep_alive_task, send_audio_task, receive_response_task)

            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                logging.info("Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    signal.signal(signal.SIGINT, assistant.handle_exit)
    logging.info("Starting Voice Assistant")
    asyncio.run(assistant.main())