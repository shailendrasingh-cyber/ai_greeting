import cv2
import numpy as np
import tkinter as tk
import pyttsx3
from PIL import Image, ImageTk

# Initialize Tkinter
root = tk.Tk()
root.title("Face Detection")

# Create a label to display the video
label = tk.Label(root)
label.pack()

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pyttsx3
engine = pyttsx3.init()

# Set properties for pyttsx3
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice
engine.setProperty('rate', 150)  # Speed of speech

def speak_message():
    engine.say("Hello, welcome to Cloud Education")
    engine.runAndWait()

def detect_face():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    face_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0 and not face_detected:
            face_detected = True
            speak_message()
        elif len(faces) == 0:
            face_detected = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert frame to ImageTk format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        root.update()

# Run the detection in a separate thread
import threading
thread = threading.Thread(target=detect_face)
thread.start()

root.mainloop()
