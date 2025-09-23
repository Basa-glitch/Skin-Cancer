import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.nn import softmax
import tkinter as tk
from tkinter import filedialog

# Load your trained model
model = load_model('C:/Users/bryan/Desktop/Skin Cancer Detection/Model/Small/student_scratch5.keras')

# Image input size expected by your model
target_size = (30, 30)

# Replace these with your actual class names
class_names = ['Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Melanoma']

# Function to choose an image
def choose_image():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title='Select an image',
        filetypes=[('Image Files', '*.jpg *.jpeg *.png')]
    )

# Predict function
def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict and apply softmax to convert logits into probabilities
    predictions = model.predict(img_array)
    probabilities = softmax(predictions[0]).numpy()
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    predicted_label = class_names[predicted_index]

    return predicted_label, confidence

# Main flow
image_path = choose_image()
if image_path:
    predicted_label, confidence = predict_image(image_path)
    expected_label = input("Enter the expected label: ")

    print(f"\n🖼️ Image: {image_path}")
    print(f"✅ Predicted: {predicted_label} ({confidence * 100:.2f}%)")
    print(f"📝 Expected : {expected_label}")
else:
    print("No image selected.")
