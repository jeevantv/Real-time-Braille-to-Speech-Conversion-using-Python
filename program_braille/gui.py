import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
from playsound import playsound
from OBR import SegmentationEngine, BrailleClassifier, BrailleImage
from gtts import gTTS

root = tk.Tk()

# Set window title
root.title("optical braille Recognition")

# Set window size
root.geometry("500x500")

# Create a canvas for the title
title_canvas = tk.Canvas(root, width=500, height=50)
title_canvas.pack()


# Global variable to store the selected image
selected_image = None
file_path = None

# Function to open file dialog and select an image
def open_image():
    global selected_image
    global file_path
    file_path = filedialog.askopenfilename()
    
    image = Image.open(file_path)
    #image = image.resize((300, 300))
    selected_image = ImageTk.PhotoImage(image)
    image_canvas.create_image(0, 0, anchor="nw", image=selected_image)
    digest_button.pack(pady=20)

# Function to perform some action on the selected image
def digest_image():
    
    classifier = BrailleClassifier()
    img = BrailleImage(file_path)
    for letters in SegmentationEngine(image=img):
        letters.mark()
        classifier.push(letters)
        classifier.digest()
    result = ""+classifier.digest()
    output_label.config(text=result)
   

def play_audio():
    classifier = BrailleClassifier()
    img = BrailleImage(file_path)
    for letters in SegmentationEngine(image=img):
        letters.mark()
        classifier.push(letters)
        classifier.digest()
   
    tts = gTTS(classifier.digest(),lang ='en', tld='co.in')
    tts.save('hello.mp3')
    audio_file = os.path.join(os.getcwd(), "hello.mp3")
    playsound(audio_file)
    os.remove(audio_file) #remove after playing 

# Create a canvas to display the selected image
image_canvas = tk.Canvas(root, width=500, height=300)
image_canvas.place(x=100, y=100)

# Create a button to trigger the file dialog
select_button = tk.Button(root, text="Select Image", command=open_image)
select_button.place(x=250, y=300)

# Create a canvas to display the "Digest" button
button_canvas = tk.Canvas(root, width=200, height=50)
button_canvas.place(x=250, y=350)

# Create a button to trigger the digest action
digest_button = tk.Button(button_canvas, text="Digest", command=digest_image)

# Create a canvas to display the output text
output_canvas = tk.Canvas(root, width=300, height=50)
output_canvas.place(x=100, y=400)

# Create a button to play audio
audio_button = tk.Button(root, text="Speech", command=play_audio)
audio_button.place(x=250, y=450)

# Create a label to display the output text
output_label = tk.Label(output_canvas, text="", font=("Helvetica", 12))
output_label.place(relx=0.5, rely=0.5, anchor="center")

# Create a canvas to display the program title
title_canvas = tk.Canvas(root, width=400, height=50)
title_canvas.place(x=50, y=25)

# Create a label to display the program title
title_label = tk.Label(title_canvas, text="Optical Braille Recognition", font=("Helvetica", 16))
title_label.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()
