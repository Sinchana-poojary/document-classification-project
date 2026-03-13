from docx2pdf import convert
import os
import time

folder_path = r"D:\Internship\dataset_balanced\new_resume"

for file in os.listdir(folder_path):

    if file.startswith("~$"):
        continue

    if file.lower().endswith(".docx"):

        doc_path = os.path.join(folder_path, file)
        pdf_path = doc_path.replace(".docx", ".pdf")

        success = False
        attempts = 0

        while not success and attempts < 3:
            try:
                convert(doc_path, pdf_path)
                print("Converted:", pdf_path)
                success = True
            except Exception as e:
                attempts += 1
                print("Retrying:", file)
                time.sleep(5)

        time.sleep(3)