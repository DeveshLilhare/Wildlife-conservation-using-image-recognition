import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('tell_model.h5')

# Create a dictionary to map class indices to animal names
class_labels = {
    0: "badger",
    1: "bat",
    2:"bear",
    3:"elephant",
    4:"tiger",
    # Add more class labels for your dataset '' '' 'bear' 'elephant' 'tiger'
}

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Open and resize the image
            image = Image.open(file_path)
            image = image.resize((224, 224))
            
            # Perform inference using the model
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array)
            class_index = np.argmax(predictions)
            class_label = class_labels.get(class_index,"Unknown")

            result_label.config(text=f"Predicted Animal: {class_label}")

            # Convert and display the image using PhotoImage
            image = ImageTk.PhotoImage(image)
            image_label = tk.Label(window, image=image)
            image_label.image = image  # Keep a reference to the image to prevent it from being garbage collected
            image_label.pack()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


window = tk.Tk()
window.geometry("600x500")
window.title("Wildlife Conservation Using Image Recognition")

# Load the background image
bgimg = ImageTk.PhotoImage(file="C:/Users/deves/Desktop/Training/First poject/1.jpg")


# Create a label to display the background image
bg_label = tk.Label(window, image=bgimg)
bg_label.place(relwidth=1, relheight=1)

# Create a label with your title
my_font1 = ('times', 18, 'bold')
title_label = tk.Label(window, text='Wildlife Conservation Using Image Recognition', font=my_font1, foreground='blue')
title_label.pack(pady=20)

my_font2 = ('times', 10 , 'bold')
title_label = tk.Label(window, text='Upload Wild Animal Image and find name is its', font=my_font1)
title_label.pack(pady=10)

my_font3 = ('times',8 , 'bold')
title_label = tk.Label(window, text='Wild Animal Image [ Bear,Bat,Badger,Tiger , Elephant ]', font=my_font1 ,foreground='green')
title_label.pack(pady=10)


# Create a label to display the result
result_label = tk.Label(window, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to upload an image and classify it
classify_button = tk.Button(window, text="Upload Image", command=classify_image, foreground='green',borderwidth=1, relief="solid")
classify_button.pack()


# Start the Tkinter main loop
window.mainloop()
