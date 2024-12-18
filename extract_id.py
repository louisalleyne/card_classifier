import cv2
import pytesseract
import os
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

# --------- MODULE 1 : Prétraitement pour images basse résolution ---------
def preprocess_low_res_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscaling avec interpolation bicubique
    upscale_factor = 2
    upscaled = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    # Filtrage bilatéral et augmentation du contraste
    denoised = cv2.bilateralFilter(upscaled, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Binarisation avec méthode d'Otsu
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image, binary, upscale_factor

# --------- MODULE 2 : Extraction de texte et encadrement ---------
def extract_text_and_boxes(binary, original_image, scale_factor):
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
    extracted_text = []

    for i in range(len(data['text'])):
        if data['text'][i].strip():
            # Ajuster les coordonnées avec le facteur d'upscaling
            x = int(data['left'][i] / scale_factor)
            y = int(data['top'][i] / scale_factor)
            w = int(data['width'][i] / scale_factor)
            h = int(data['height'][i] / scale_factor)

            # Ajouter le texte détecté
            extracted_text.append(data['text'][i].strip())

            # Dessiner les rectangles et le texte ajusté sur l'image originale
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_image, data['text'][i].strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return extracted_text, original_image

# --------- MODULE 3 : Redimensionnement pour affichage ---------
def resize_for_display(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    scaling_factor = min(max_width / w, max_height / h)
    new_dimensions = (int(w * scaling_factor), int(h * scaling_factor))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

# --------- MODULE 4 : Pipeline principale ---------
def extract_id(image_path):

  if os.path.isfile(image_path):
      print(f"\n--- Traitement de l'image : {image_path} ---")
      original_image, processed_image, scale_factor = preprocess_low_res_image(image_path)
      extracted_text, annotated_image = extract_text_and_boxes(processed_image, original_image, scale_factor)

      # Afficher le texte extrait
      print("Texte extrait :\n", "\n".join(extracted_text))

      # Redimensionner les images pour l'affichage
      resized_annotated = resize_for_display(annotated_image)
      resized_processed = resize_for_display(processed_image)

      # Afficher les images
      cv2.imshow("Image Annotée Redimensionnée", resized_annotated)
      cv2.waitKey(0)
      cv2.destroyAllWindows()