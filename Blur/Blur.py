import torch
import sys
import os
from pathlib import Path
import cv2
import subprocess
import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.utils import get_array_type
import tkinter as tk
from tkinter import ttk





from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes

model_path = "./best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetectMultiBackend(model_path, device=device, dnn=False, data='./coco128.yaml', fp16=False)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Не удалось получить доступ к камере")
    sys.exit()

video_ffmpeg_process = subprocess.Popen([
    'ffmpeg',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '640x640',  
    '-r', '30',  
    '-i', '-',  
    '-f', 'v4l2',
    '/dev/video2'  
], stdin=subprocess.PIPE)


RATE = 44100
CHUNK = 1024
CHANNELS = 1
SEMITONES = -4  


def pitch_shift(data, rate, semitones):
    sound = AudioSegment(
        data.tobytes(), 
        frame_rate=rate,
        sample_width=data.dtype.itemsize, 
        channels=data.shape[1] if len(data.shape) > 1 else 1
    )
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    pitch_shifted_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    pitch_shifted_sound = pitch_shifted_sound.set_frame_rate(rate)
    array_type = get_array_type(pitch_shifted_sound.sample_width * 8)
    pitch_shifted_array = np.array(pitch_shifted_sound.get_array_of_samples(), dtype=array_type)
    

    if len(pitch_shifted_array) > len(data):
        pitch_shifted_array = pitch_shifted_array[:len(data)]
    elif len(pitch_shifted_array) < len(data):
        pitch_shifted_array = np.pad(pitch_shifted_array, (0, len(data) - len(pitch_shifted_array)), mode='constant')

    return pitch_shifted_array.reshape(data.shape)


audio_ffmpeg_process = subprocess.Popen([
    'ffmpeg',
    '-f', 's16le',
    '-ar', str(RATE),
    '-ac', str(CHANNELS),
    '-i', '-',  
    '-f', 'alsa',
    'hw:Loopback,1,0' 
], stdin=subprocess.PIPE)


p = pyaudio.PyAudio()


global semitones
semitones = -4

def audio_callback(in_data, frame_count, time_info, status):
    global semitones
    data = np.frombuffer(in_data, dtype=np.int16).reshape(-1, CHANNELS)
    out_data = pitch_shift(data, RATE, semitones)
    try:
        audio_ffmpeg_process.stdin.write(out_data.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe is broken.")
        return (None, pyaudio.paComplete)
    return (in_data, pyaudio.paContinue)

stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)


def update_semitones(val):
    global semitones
    semitones = int(val)


root = tk.Tk()
root.title("Pitch Shifter")

ttk.Label(root, text="Pitch Shift (semitones)").pack(pady=10)
pitch_slider = ttk.Scale(root, from_=-10, to=10, orient='horizontal', command=update_semitones)
pitch_slider.set(semitones)
pitch_slider.pack(pady=20)

def on_closing():
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_ffmpeg_process.stdin.close()
    audio_ffmpeg_process.wait()
    video_ffmpeg_process.stdin.close()
    video_ffmpeg_process.wait()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

print("Recording and processing... Press Ctrl+C to stop.")
stream.start_stream()


def process_video():
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить изображение с камеры")
            break

        img_resized = cv2.resize(frame, (640, 640))  
        img_transposed = img_resized.transpose(2, 0, 1)  
        img_tensor = torch.from_numpy(img_transposed).to(device)
        img_tensor = img_tensor.float() / 255.0  
        img_tensor = img_tensor.unsqueeze(0)  

        with torch.no_grad():
            pred = model(img_tensor)

        pred = non_max_suppression(pred)  

        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in det:
                    xyxy = [int(x) for x in xyxy]  
                    x1, y1, x2, y2 = xyxy

                    x1 = max(0, min(x1, img_resized.shape[1] - 1))
                    y1 = max(0, min(y1, img_resized.shape[0] - 1))
                    x2 = max(0, min(x2, img_resized.shape[1] - 1))
                    y2 = max(0, min(y2, img_resized.shape[0] - 1))

                    roi = img_resized[y1:y2, x1:x2]

                    if roi.size == 0:
                        continue

                    roi = cv2.GaussianBlur(roi, (51, 51), 30)

                    img_resized[y1:y2, x1:x2] = roi

        try:
            video_ffmpeg_process.stdin.write(img_resized.tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe is broken.")
            break

        frame_count += 1
        print(f"Processed frame {frame_count}")

        cv2.imshow('Video with Blurred Bounding Boxes', img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


import threading
video_thread = threading.Thread(target=process_video)
video_thread.start()

root.mainloop()
