import os
from pdf2image import convert_from_path

folder_path = r"D:\Internship\dataset_balanced\new_resume"
poppler_path = r"C:\poppler-23.11.0\Library\bin"

for file in os.listdir(folder_path):

    if file.lower().endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        pages = convert_from_path(pdf_path, poppler_path=poppler_path)

        for i, page in enumerate(pages):

            image_name = file.replace(".pdf", f"_page{i+1}.jpg")
            save_path = os.path.join(folder_path, image_name)

            page.save(save_path, "JPEG")

            print("Saved:", image_name)