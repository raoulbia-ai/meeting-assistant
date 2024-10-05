import pyaudio
import wave
import struct
import numpy as np

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "test_audio.wav"

def record_audio():
    """
    Function to record audio from the microphone and save to a .wav file
    """
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording audio")

    frames = []

    # Record for RECORD_SECONDS
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Recording finished")

    # Stop and close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio as a .wav file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")

def play_audio():
    """
    Function to play back the recorded audio
    """
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')

    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play back the audio file in chunks
    data = wf.readframes(CHUNK)

    print("* Playing back audio")
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # Stop and close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("* Playback finished")

if __name__ == "__main__":
    record_audio()  # Record audio for 5 seconds
    play_audio()    # Play back the recorded audio
