import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import subprocess
import playsound
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage
from gtts import gTTS
import os

# Global variables
selected_image = None
file_path = None
audio_file = "./hello.mp3"

def open_image_window():
    image_window = tk.Toplevel(root)
    image_window.geometry("500x500")
    image_window.title("Selected Image")

    canvas = tk.Canvas(image_window)
    canvas.pack(fill="both", expand=True)

    image = Image.open(file_path)
    image = image.resize((400, 400))
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(250, 250, image=image_tk, anchor="center")
    canvas.image = image_tk

    # Create "Digest" button
    digest_button = tk.Button(image_window, text="Digest", command=perform_action)
    digest_button.pack(pady=10)

    # Create "Speech" button
    speech_button = tk.Button(image_window, text="Speech", command=play_audio)
    speech_button.pack(pady=10)

    # Create output label
    global output_label
    output_label = tk.Label(image_window, text="", wraplength=400)
    output_label.pack()

def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()

    if file_path:
        open_image_window()

def perform_action():
    # Replace with your desired action
    classifier = BrailleClassifier()
    img = BrailleImage(file_path)
    for letters in SegmentationEngine(image=img):
        letters.mark()
        classifier.push(letters)
        classifier.digest()
    result = ""+classifier.digest()
    output_label.config(text=result)

    tts = gTTS(classifier.digest(),lang ='en', tld='co.in')
    tts.save('hello.mp3')

def execute_python_file():
    # Replace "path/to/your/file.py" with the actual path to your Python file
    file_path = "./real_time.py"
    # Execute the Python file and capture the output
    output = subprocess.check_output(["python", file_path]).decode("utf-8")
    # Display the output in a label
    output_label.config(text=output)

def play_audio():
    global audio_file
    playsound.playsound(audio_file)
    os.remove(audio_file)
    

def open_realtime_window():
    realtime_window = tk.Toplevel(root)
    realtime_window.geometry("500x500")
    realtime_window.title("Real-time Conversion")

    # Create a label for real-time conversion
    label = tk.Label(realtime_window, text="Real-time Conversion")
    label.pack()

    # Add "Start" button to initiate real-time conversion
    button = tk.Button(realtime_window, text="Start", command=execute_python_file)
    button.pack()

    # Create a label to display the output
    global output_label
    output_label = tk.Label(realtime_window, text="", wraplength=400)
    output_label.pack()

    # Add "Play Audio" button
    global play_button
    play_button = tk.Button(realtime_window, text="Play Audio", command=play_audio)
    play_button.pack()

# Create the main window
root = tk.Tk()
root.geometry("700x700")
root.title("Real-time Braille to Speech Conversion")

# Create the first canvas on top
canvas1 = tk.Canvas(root)
canvas1.pack(fill="both", expand=True)

# Set the options text on the first canvas
options_text = """
Real-time Braille to Speech Conversion using Python

Choose any one from below:
"""

# Set the title on the first canvas
canvas1.create_text(350, 100, text=options_text,
                    font=("Arial", 16), fill="black", anchor="center")

# Create canvas2 on the left side
canvas2 = tk.Canvas(root, width=350, height=700)
canvas2.pack(side="left")

# Add "Upload Image" button to canvas2
upload_image_button = tk.Button(canvas2, text="Upload Image", command=upload_image)
upload_image_button.place(relx=0.5, rely=0.5, anchor="center")

# Create canvas3 on the right side
canvas3 = tk.Canvas(root, width=350, height=700)
canvas3.pack(side="left")

# Add "Real-time" button to canvas3
real_time_button = tk.Button(canvas3, text="Real-time", command=open_realtime_window)
real_time_button.place(relx=0.5, rely=0.5, anchor="center")

# Start the GUI event loop
root.mainloop()
