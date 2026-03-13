# document-classification-project

Document Classification System
A deep learning–based document classification system that automatically identifies different types of documents such as Resume, Invoice, Salary Slip, Aadhar card and PAN card from images or PDF files.
The model is trained using TensorFlow/Keras and processes document images to predict their category.

Features
Classifies multiple document types automatically
Supports both image files and PDF documents
Converts PDF files to images before prediction
Uses a deep learning classification model
Includes confidence-based detection for unknown documents ("other")
Allows testing a single document or an entire folder of documents

Supported Document Classes
The system currently classifies the following document types:
Resume
Invoice
Salary Slip
Aadhar card
PAN card

If the prediction confidence is low, the system outputs:other

Project Structure
Internship/
│
├── dataset_balanced/
│   ├── resume/
│   ├── invoice/
│   ├── salary_slip/
│   ├── aadhar_card/
│   ├── pan_card/
│
├── test_documents/
│
├── train_model.py
├── predict_document.py
├── test_folder_prediction.py
│
├── document_classifier_model.h5
│
└── README.md

Installation
Install the required dependencies:
pip install tensorflow numpy pillow pdf2image scikit-learn pytesseract

Additional Requirements
The project requires Poppler for converting PDF files to images.
Download Poppler for Windows:
https://github.com/oschwartz10612/poppler-windows/releases

Extract the folder and update the path in the code:
poppler_path = r"C:\poppler\Library\bin"

Dataset Preparation
The dataset must be organized in the following structure:
dataset_balanced
   resume
   invoice
   salary_slip
   aadhar_card
   pan_card
Each folder should contain only image files (.jpg or .png).

Training the Model
Run the training script:
python train_model.py

After training, the model will be saved as:
document_classifier_model.h5

Predicting a Single Document
Run the prediction script:
python predict_document.py
Example output:
resume : 0.82
invoice : 0.10
salary_slip : 0.05
id_proof : 0.03
Final Prediction: resume
If the model is not confident:
Final Prediction: other

Testing Multiple Documents
Place multiple files inside the folder:
test_documents/
Run:
python test_folder_prediction.py
Example output:
File: resume1.jpg
Prediction: resume
File: invoice_sample.jpg
Prediction: invoice
File: random_document.jpg
Prediction: other

Technologies Used
Python
TensorFlow / Keras
NumPy
Pillow
pdf2image