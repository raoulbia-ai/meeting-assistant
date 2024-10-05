import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

print("Available input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels') > 0:
        print(f"Input Device id {i} - {dev.get('name')}")

device_id = int(input("Enter input device ID to use: "))

# Get device info
device_info = p.get_device_info_by_index(device_id)
print(f"Selected device: {device_info['name']}")

# Get supported sample rates
supported_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
for rate in supported_rates:
    try:
        if p.is_format_supported(rate, input_device=device_id, input_channels=CHANNELS, input_format=FORMAT):
            print(f"Supported sample rate: {rate}")
    except:
        pass

# Ask user to choose a supported rate
RATE = int(input("Enter a supported sample rate from the list above: "))

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    p.terminate()