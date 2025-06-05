import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Fix encoding issue in Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model_path = r'C:\Users\imroz\Music\skin problem detect\skin_problem_model.h5'
model = load_model(model_path)

# Define the labels
labels = ['dark_circles', 'enlarged_pores', 'fine_lines', 'hyperpigmentation', 'pimples', 'wrinkles']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Press 's' to capture", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        img_path = 'captured_image.jpg'
        cv2.imwrite(img_path, frame)

        # Preprocess
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)

        # Output prediction
        print("Predictions:")
        for i, label in enumerate(labels):
            try:
                print(f"{label}: {predictions[0][i] * 100:.2f}%")
            except UnicodeEncodeError:
                print(label.encode('ascii', errors='ignore').decode(), f": {predictions[0][i] * 100:.2f}%")

        # Annotate on image
        cv2.putText(frame, "Prediction Results", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset = 60
        for i, label in enumerate(labels):
            text = f"{label}: {predictions[0][i] * 100:.2f}%"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        cv2.imshow("Predicted Image", frame)
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()
