import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("waste_segregation_model.h5")

# Define class labels (same order as training)
labels = ["glass", "metal", "paper", "plastic"]

# -----------------------------
# Prediction function
# -----------------------------
def gradio_predict(img):
    img = cv2.resize(img, (128, 128))   # resize to match training
    img = img / 255.0                   # normalize
    img = np.expand_dims(img, axis=0)   # add batch dimension

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {labels[class_idx]: confidence}

# -----------------------------
# Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4),
    title="♻️ Waste Segregation CNN",
    description="Upload an image of waste (glass, metal, paper, plastic) and the model will classify it."
)

demo.launch()
