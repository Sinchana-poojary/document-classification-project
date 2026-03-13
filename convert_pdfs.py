from pdf2image import convert_from_path
import os

dataset_path = "dataset_balanced"
poppler_path = r"C:\poppler-23.11.0\Library\bin"

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".pdf"):


            pdf_path = os.path.join(root, file)

            pages = convert_from_path(pdf_path, poppler_path=poppler_path)

            for i, page in enumerate(pages):

                img_name = file.replace(".pdf", f"_{i}.jpg")
                img_path = os.path.join(root, img_name)

                page.save(img_path, "JPEG")

            print("Converted:", file)