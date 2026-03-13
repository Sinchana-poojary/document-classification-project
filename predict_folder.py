import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# load model
model = tf.keras.models.load_model("document_classifier_model.h5", compile=False)

class_names = ['aadhar_card', 'invoice', 'pan_card', 'resume', 'salary_slip']

# folder containing test documents
folder_path = r"D:\Internship\test_documents"

poppler_path = r"C:\poppler-23.11.0\Library\bin"

threshold = 0.6

for file in os.listdir(folder_path):

    file_path = os.path.join(folder_path, file)

    if file.lower().endswith((".jpg",".jpeg",".png",".pdf")):

        # load image
        if file.lower().endswith(".pdf"):
            pages = convert_from_path(file_path, poppler_path=poppler_path)
            img = pages[0]
        else:
            img = Image.open(file_path)

        img = img.convert("RGB")
        img = img.resize((224,224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]

        max_index = np.argmax(prediction)
        confidence = prediction[max_index]

        if confidence < threshold:
            predicted_class = "other"
        else:
            predicted_class = class_names[max_index]

        print("\nFile:", file)
        print("Confidence:", round(confidence,4))
        print("Prediction:", predicted_class)