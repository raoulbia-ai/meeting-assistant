import os
import pyaudio
import json
import websockets
import asyncio
from dotenv import load_dotenv
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    def handle_exit(self, signal, frame):
        logger.info("Exiting the application.")
        sys.exit(0)

    def generate_event_id(self):
        return f"event_{uuid.uuid4().hex[:6]}"

    async def initialize_session(self, websocket):
        session_update = {
            "event_id": self.generate_event_id(),
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful assistant. Your knowledge cutoff is 2023-10.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "temperature": 0.8
            }
        }
        
        logger.debug(f"Sending session update request: {json.dumps(session_update)}")
        await websocket.send(json.dumps(session_update))
        
        for attempt in range(5):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                if response_data.get("type") in ["session.created", "session.updated"]:
                    logger.info(f"Session {response_data.get('type')} successfully.")
                    return True
                elif response_data.get("type") == "error":
                    error_data = response_data.get('error', {})
                    logger.error(f"Error initializing session: {error_data.get('type')} - {error_data.get('code')} - {error_data.get('message')}")
                    return False
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for session.")
                await asyncio.sleep(1)
        logger.error("Failed to initialize session after multiple attempts.")
        return False

    def select_audio_device(self):
        p = pyaudio.PyAudio()
        logger.info("Listing available audio devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            logger.info(f"Device {i}: {device_info['name']}")
        
        device_index = int(input("Enter the device index for your microphone: "))
        return device_index

    async def stream_audio_to_api(self, websocket):
        p = pyaudio.PyAudio()
        device_index = self.select_audio_device()

        try:
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)
        except Exception as e:
            logger.error(f"Failed to open stream: {str(e)}")
            return

        logger.info(f"Streaming audio to API at {RATE} Hz...")

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

        except Exception as e:
            logger.error(f"Error in sending audio data: {str(e)}")
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

            logger.debug(f"Sending audio chunk: {len(self.audio_buffer)} bytes")

            commit_message = {
                "event_id": self.generate_event_id(),
                "type": "input_audio_buffer.commit"
            }
            await websocket.send(json.dumps(commit_message))
            
            self.audio_buffer = b""
        else:
            logger.debug("Audio buffer is empty. Skipping send.")

    def process_audio_delta(self, message):
        audio_data = base64.b64decode(message.get('audio', ''))
        logger.debug(f"Received {len(audio_data)} bytes of audio data")
        # Here you would typically send this to an audio player

    def process_transcript_delta(self, message):
        transcript_delta = message.get('delta', '')
        self.full_transcript += transcript_delta
        logger.info(f"Transcript delta: {transcript_delta}")
        logger.info(f"Full transcript: {self.full_transcript}")

    async def openai_realtime_api_interaction(self, websocket):
        try:
            while True:
                response = await websocket.recv()
                if isinstance(response, str):
                    try:
                        message = json.loads(response)
                        if message['type'] == 'response.audio.delta':
                            self.process_audio_delta(message)
                        elif message['type'] == 'response.audio_transcript.delta':
                            self.process_transcript_delta(message)
                        elif message['type'] == 'error':
                            logger.error(f"Error from API: {json.dumps(message)}")
                        else:
                            logger.debug(f"Received message type: {message['type']}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {response}")
                else:
                    logger.warning(f"Received non-string message: {type(response)}")
        except Exception as e:
            logger.error(f"Error in receiving response: {str(e)}")

    async def keep_alive(self, websocket):
        while True:
            try:
                await websocket.ping()
                await asyncio.sleep(20)
            except:
                break

    async def main(self):
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set. Please check your .env file.")
            return

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to {OPENAI_REALTIME_API_URL} (attempt {attempt + 1}/{max_retries})")
                async with websockets.connect(OPENAI_REALTIME_API_URL, extra_headers=headers, timeout=30) as websocket:
                    logger.info("Connected to Realtime API")

                    logger.info("Initializing session...")
                    if not await self.initialize_session(websocket):
                        logger.error("Failed to initialize session. Exiting.")
                        return

                    logger.info("Session initialized successfully. Starting audio streaming...")

                    keep_alive_task = asyncio.create_task(self.keep_alive(websocket))
                    send_audio_task = asyncio.create_task(self.stream_audio_to_api(websocket))
                    receive_response_task = asyncio.create_task(self.openai_realtime_api_interaction(websocket))

                    await asyncio.gather(keep_alive_task, send_audio_task, receive_response_task)

            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Exiting.")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                break
            finally:
                logger.info("Closing WebSocket connection...")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    signal.signal(signal.SIGINT, assistant.handle_exit)
    asyncio.run(assistant.main())