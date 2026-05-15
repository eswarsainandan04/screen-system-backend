import fitz
import os

# ==========================================
# CONFIG
# ==========================================

# PDF_PATH = "../knowledge_base/AIML_role/Machine_Learning_For_Absolute_Beginners.pdf"
PDF_PATH = "../knowledge_base/Data_Science_Applied_ML/Introduction_to_Machine_Learning_with_Python.pdf"

TEXT_OUTPUT_DIR = "../extracted_text"

IMAGE_OUTPUT_DIR = "../extracted_images/ds_applied_ml"

TEXT_OUTPUT_FILE = "ds_book.txt"

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# ==========================================
# OPEN PDF
# ==========================================

doc = fitz.open(PDF_PATH)

print(f"\nOpened PDF with {len(doc)} pages")

# ==========================================
# EXTRACT TEXT + IMAGES
# ==========================================

full_text = ""

image_counter = 0

for page_index in range(len(doc)):

    page = doc[page_index]

    print(f"\nProcessing Page {page_index + 1}")

    # --------------------------------------
    # TEXT EXTRACTION
    # --------------------------------------

    page_text = page.get_text()

    if page_text:
        full_text += f"\n\n===== PAGE {page_index + 1} =====\n\n"
        full_text += page_text

    # --------------------------------------
    # IMAGE EXTRACTION
    # --------------------------------------

    images = page.get_images(full=True)

    print(f"Found {len(images)} images")

    for img_index, img in enumerate(images):

        xref = img[0]

        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]

        image_ext = base_image["ext"]

        # ----------------------------------
        # SAVE IMAGE
        # ----------------------------------

        image_filename = (
            f"page_{page_index + 1}_img_{img_index + 1}.{image_ext}"
        )

        image_path = os.path.join(
            IMAGE_OUTPUT_DIR,
            image_filename
        )

        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)

        print(f"Saved image: {image_filename}")

        # ----------------------------------
        # ADD PLACEHOLDER TO TEXT
        # ----------------------------------

        full_text += (
            f"\n[IMAGE: {image_filename} extracted from page "
            f"{page_index + 1}]\n"
        )

        image_counter += 1

# ==========================================
# SAVE TEXT FILE
# ==========================================

output_text_path = os.path.join(
    TEXT_OUTPUT_DIR,
    TEXT_OUTPUT_FILE
)

with open(output_text_path, "w", encoding="utf-8") as file:
    file.write(full_text)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")
print("PDF Processing Complete")
print("===================================")

print(f"Total Pages: {len(doc)}")
print(f"Total Images Extracted: {image_counter}")

print(f"\nText saved to:\n{output_text_path}")

print(f"\nImages saved to:\n{IMAGE_OUTPUT_DIR}")