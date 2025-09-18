import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import cv2

# -----------------------------
# 1. DATASET PATH
# -----------------------------
dataset_path = '/Users/devrajbhandarkar/Desktop/Project/dataset'  # Your dataset folder
img_size = (128, 128)     # Resize all images
batch_size = 32

# -----------------------------
# 2. DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

print("Class Mapping:", train_data.class_indices)

# -----------------------------
# 3. MODEL (CNN)
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")  # 4 classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# -----------------------------
# 4. TRAINING
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

# -----------------------------
# 5. TRAINING PERFORMANCE PLOTS
# -----------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Model Loss")
plt.show()

# -----------------------------
# 6. PREDICTION FUNCTION
# -----------------------------
labels = list(train_data.class_indices.keys())

def predict_waste(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return labels[class_idx], confidence

# -----------------------------
# 7. TEST PREDICTION
# -----------------------------
test_img = '/Users/devrajbhandarkar/Desktop/Project/validation'  # put your test image path here
pred_class, conf = predict_waste(test_img)
print(f"Predicted Waste Type: {pred_class} (Confidence: {conf:.2f})")

# -----------------------------
# 8. SAVE MODEL
# -----------------------------
model.save("waste_segregation_model.h5")
print("Model saved as waste_segregation_model.h5")
