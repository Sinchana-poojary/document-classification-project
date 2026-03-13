import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

model = tf.keras.models.load_model("document_classifier_model.h5", compile=False)

class_names = ['id_proof','invoice','resume','salary_slip']

file_path = r"D:\Internship\test_documents\doc2.jpg"

poppler_path = r"C:\poppler-23.11.0\Library\bin"

if file_path.lower().endswith(".pdf"):
    pages = convert_from_path(file_path, poppler_path=poppler_path)
    img = pages[0]
else:
    img = Image.open(file_path)

img = img.convert("RGB")
img = img.resize((224,224))

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array, verbose=0)[0]

for i, prob in enumerate(prediction):
    print(class_names[i], ":", prob)

predicted_class = class_names[np.argmax(prediction)]

print("\nFinal Prediction:", predicted_class)