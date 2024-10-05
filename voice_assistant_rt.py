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

# Load environment variables
load_dotenv()

# Set OpenAI API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio settings
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Start with 44100 Hz (common sample rate)

# Setup logging
logging.basicConfig(filename='/var/log/voice_assistant.log', level=logging.DEBUG)

# Debug mode flag
DEBUG = True

def test_audio_capture(p, device_index, rate):
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
        
        print("Testing audio capture...")
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        if len(audio_data) > 0:
            print(f"Successfully captured {len(audio_data)} bytes of audio data.")
        else:
            print("Failed to capture audio data. Please check your microphone settings.")
        
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error testing audio capture: {str(e)}")


# Handle graceful shutdown on Ctrl+C
def handle_exit(signal, frame):
    print("\nExiting the application.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

async def openai_realtime_api_interaction(websocket):
    print("Waiting for API response...")
    logging.debug("Waiting for API response...")
    transcription = ""
    
    try:
        while True:
            response = await websocket.recv()
            if DEBUG:
                logging.debug(f"Received raw response: {response}")
                print(f"Received raw response: {response}")

            response_data = json.loads(response)
            
            if response_data.get("type") == "response.text.delta":
                transcription += response_data.get("delta", {}).get("text", "")
                print(f"Transcription so far: {transcription}")
                logging.debug(f"Transcription so far: {transcription}")

            elif response_data.get("type") == "response.audio.delta":
                # Handle audio response
                audio_data = base64.b64decode(response_data.get("delta", {}).get("audio", ""))
                # Process audio_data (e.g., play it or save it)
                print("Received audio response")
                logging.debug("Received audio response")

            elif response_data.get("type") == "error":
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                print(f"Error from API: {error_message}")
                logging.error(f"Error from API: {error_message}")

            elif response_data.get("type") == "response.end":
                print("End of response.")
                logging.debug("End of response.")
                break

            else:
                logging.warning(f"Unexpected message type received: {response_data}")
                print(f"Unexpected message type received: {response_data}")

    except Exception as e:
        print(f"Error in receiving response: {str(e)}")
        logging.error(f"Error in receiving response: {str(e)}")

    return transcription

async def stream_audio_to_api(websocket):
    global RATE
    p = pyaudio.PyAudio()

    # List audio devices for debugging
    print("Listing available audio devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']}")

    # Use device index 7 by default (C922 Pro Stream Webcam: USB Audio)
    device_index = 7

    # Call this function before starting the main loop
    test_audio_capture(p, device_index, RATE)

    # Attempt to open the audio stream with the specified sample rate
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Failed to open stream with rate {RATE}. Retrying with 24000 Hz.")
        RATE = 24000  # Fallback to 24000 Hz
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)

    print(f"* Streaming audio to API at {RATE} Hz...")
    logging.debug("* Streaming audio to API...")

    try:
        while True:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            if len(audio_data) == 0:
                print("No audio data captured. Skipping this chunk.")
                continue
            # Ensure audio is captured and base64-encoded
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")
            if not encoded_audio:
                print("Failed to encode audio data. Skipping this chunk.")
                continue

            append_message = {
                "type": "input_audio_buffer.append",
                "input": {
                    "audio": encoded_audio,
                    "format": "pcm16",
                    "rate": RATE
                }
            }
            await websocket.send(json.dumps(append_message))

            # Only commit if we successfully sent audio data
            commit_message = {"type": "input_audio_buffer.commit"}
            await websocket.send(json.dumps(commit_message))

            logging.debug(f"Audio data length: {len(audio_data)}")
            logging.debug(f"Encoded audio length: {len(encoded_audio)}")

            logging.debug(f"Sending audio chunk: {len(audio_data)} bytes.")
            await websocket.send(json.dumps(append_message))

            # Commit the buffer
            commit_message = {"type": "input_audio_buffer.commit"}
            await websocket.send(json.dumps(commit_message))

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error in sending audio data: {str(e)}")
        logging.error(f"Error in sending audio data: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def keep_alive(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(20)
        except:
            break

async def initialize_session(websocket):
    session_update = {
        "type": "session.update",
        "input_audio_transcription": {
            "enabled": True,
            "model": "whisper-1"
        }
    }
    
    logging.debug(f"Sending session update request: {json.dumps(session_update)}")
    await websocket.send(json.dumps(session_update))
    
    # Wait for confirmation
    for attempt in range(5):  # Try up to 5 times
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            if response_data.get("type") == "session.created":
                print(f"Session created successfully.")
                return True
            elif response_data.get("type") == "error":
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                print(f"Error initializing session: {error_message}")
                logging.error(f"Error initializing session: {error_message}")
                return False
        except asyncio.TimeoutError:
            print("Timeout waiting for session.")
            logging.warning("Timeout waiting for session.")
            await asyncio.sleep(1)
    print("Failed to initialize session after multiple attempts.")
    logging.error("Failed to initialize session after multiple attempts.")
    return False

async def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set. Please check your .env file.")
        logging.error("OPENAI_API_KEY is not set.")
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to {OPENAI_REALTIME_API_URL} (attempt {attempt + 1}/{max_retries})")
            async with websockets.connect(OPENAI_REALTIME_API_URL, extra_headers=headers, timeout=30) as websocket:
                print("Connected to Realtime API")
                logging.info("Connected to Realtime API")

                print("Initializing session...")
                if not await initialize_session(websocket):
                    print(f"Failed to initialize session (attempt {attempt + 1}/{max_retries}). Retrying...")
                    logging.error(f"Failed to initialize session (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue

                print("Session initialized successfully. Starting audio streaming...")
                logging.info("Session initialized successfully. Starting audio streaming...")

                keep_alive_task = asyncio.create_task(keep_alive(websocket))
                send_audio_task = asyncio.create_task(stream_audio_to_api(websocket))
                receive_response_task = asyncio.create_task(openai_realtime_api_interaction(websocket))

                await asyncio.gather(keep_alive_task, send_audio_task, receive_response_task)

        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            print(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            logging.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries reached. Exiting.")
                logging.error("Max retries reached. Exiting.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            logging.error(f"Unexpected error: {str(e)}")
            break
        finally:
            print("Closing WebSocket connection...")
            logging.debug("Closing WebSocket connection...")

if __name__ == "__main__":
    asyncio.run(main())
