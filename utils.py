import cv2
import numpy as np
import pytesseract
from scipy.stats import entropy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import logging
import os
from dotenv import load_dotenv
from extract_id import extract_id

load_dotenv()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("utils.log"),
    ]
)

logger = logging.getLogger("UtilsLogger")

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

def find_faces(image_path):
    ksize = (50, 50)

    img = cv2.imread(image_path)
    image = cv2.bilateralFilter(img, 10, 75, 75)
    #image = cv2.blur(image, ksize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    min_size = int(img.shape[1] / 10)

    faces = face_classifier.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )

    return faces

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def extract_image_characteristics(image_path):
    logger.info(f"Chargement de l'image depuis : {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Impossible de charger l'image depuis {image_path}")
        raise FileNotFoundError(f"Impossible de charger l'image depuis {image_path}")
    
    logger.info("Image chargée avec succès.")
    characteristics = {}
    
    logger.info("Conversion de l'image en niveaux de gris.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    logger.info(f"Résolution détectée : largeur={width}, hauteur={height}")
    characteristics['resolution'] = (width, height)
    characteristics['aspect_ratio'] = width / height

    logger.info("Calcul du niveau de bruit.")
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    characteristics['noise_level'] = noise_level
    logger.info(f"Niveau de bruit : {noise_level:.2f}")

    logger.info("Calcul de la luminosité et du contraste.")
    brightness = np.mean(gray)
    contrast = gray.std()
    characteristics['brightness'] = brightness
    characteristics['contrast'] = contrast
    logger.info(f"Luminosité : {brightness:.2f}, Contraste : {contrast:.2f}")

    logger.info("Calcul de l'entropie de l'histogramme.")
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_entropy = entropy(hist + 1e-9)
    characteristics['histogram_entropy'] = hist_entropy
    logger.info(f"Entropie de l'histogramme : {hist_entropy:.2f}")

    logger.info("Détection des lignes pour déterminer l'angle d'inclinaison.")
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = [np.degrees(line[0][1]) - 90 for line in lines]
        skew_angle = np.median(angles)
        logger.info(f"Angle d'inclinaison détecté : {skew_angle:.2f}°")
    else:
        skew_angle = 0
        logger.info("Aucun angle d'inclinaison détecté.")
    characteristics['skew_angle'] = skew_angle

    logger.info("Calcul de la densité de texte.")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_density = len(contours) / (width * height)
    characteristics['text_density'] = text_density
    logger.info(f"Densité de texte : {text_density:.5f}")

    logger.info("Détection du mode de couleur.")
    unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    if unique_colors == 1:
        color_mode = 'grayscale'
    else:
        color_mode = 'color'
    characteristics['color_mode'] = color_mode
    logger.info(f"Mode de couleur : {color_mode}")

    logger.info("Calcul du rapport de fond blanc.")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    characteristics['background_white_ratio'] = white_pixels / (white_pixels + black_pixels)
    logger.info(f"Rapport fond blanc : {characteristics['background_white_ratio']:.5f}")

    logger.info("Analyse de l'orientation du texte.")
    horizontal_lines = np.sum(np.mean(edges, axis=0) > 0.1)
    vertical_lines = np.sum(np.mean(edges, axis=1) > 0.1)
    orientation = 'horizontal' if horizontal_lines > vertical_lines else 'vertical'
    characteristics['text_orientation'] = orientation
    logger.info(f"Orientation du texte : {orientation}")

    logger.info("Extraction des caractéristiques terminée.")
    return characteristics

def extract_params(characteristics):
    
    bilateral = 10
    equal_hist = True
    blur = True

    if characteristics['noise_level'] < 250:
      blur = False
      bilateral = 5
  
    if characteristics['background_white_ratio'] < 0.4:
      equal_hist = False
    
    if characteristics['brightness'] < 120:
      binary = 65
    elif characteristics['brightness'] < 140:
      binary = 10
    else:
      binary = 30

    params = {
        "ksize": (3, 3),
        "kernel_size": (1, 4),
        "iterations": 1,
        "bilateralFilter": [
            bilateral,
            75,
            75
        ],
        "blur": blur,
        "sharpen": True,
        "equal_hist": equal_hist,
        "binary": binary,
        "morphology": []
    }

    return params

def check_keys_found(obj):
    for _,value in obj.items():
        if value == None:
            return False
    return True

def extract_student_card(image, student):
    text_boxes = pytesseract.image_to_data(image, lang='fra', output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(image, lang='fra')

    name_regex = r"\b(?!(ETUDIANT|UDIANT)\b)[A-Z]{3,}\s+(?!(ETUDIANT|UDIANT)\b)[A-Z]{3,}\b"
    year_regex = r"\d{2}\/\d{2}"
    ine_regex = r"\b(\d{10})\s*([a-z])\b"
    student_num_regex = r"\b\d{6}\b"

    name_match = re.search(name_regex, text)
    year_match = re.search(year_regex, text)
    ine_match = re.search(ine_regex, text)
    student_num_match = re.search(student_num_regex, text)

    if student.get("fname") is None and name_match:
        name_parts = name_match.group().replace("\n", " ").split()
        fname = name_parts[1]
        try:
            word_index = text_boxes["text"].index(fname)
            x, y, w, h = (
                text_boxes["left"][word_index],
                text_boxes["top"][word_index],
                text_boxes["width"][word_index],
                text_boxes["height"][word_index],
            )
            student["fname"] = {"value": fname, "coords": [x, y, w, h]}
            logger.info(f"Prénom extrait: {fname} avec coordonnées: {x}, {y}, {w}, {h}")
        except ValueError:
            student["fname"] = {"value": fname, "coords": None}
            logger.warning(f"Le prénom {fname} n'a pas pu être localisé dans l'image.")

    if student.get("lname") is None and name_match:
        name_parts = name_match.group().replace("\n", " ").split()
        lname = name_parts[0]
        try:
            word_index = text_boxes["text"].index(lname)
            x, y, w, h = (
                text_boxes["left"][word_index],
                text_boxes["top"][word_index],
                text_boxes["width"][word_index],
                text_boxes["height"][word_index],
            )
            student["lname"] = {"value": lname, "coords": [x, y, w, h]}
            logger.info(f"Nom extrait: {lname} avec coordonnées: {x}, {y}, {w}, {h}")
        except ValueError:
            student["lname"] = {"value": lname, "coords": None}
            logger.warning(f"Le nom {lname} n'a pas pu être localisé dans l'image.")

    if student.get("year") is None and year_match:
        year = year_match.group()
        try:
            word_index = text_boxes["text"].index(year)
            x, y, w, h = (
                text_boxes["left"][word_index],
                text_boxes["top"][word_index],
                text_boxes["width"][word_index],
                text_boxes["height"][word_index],
            )
            student["year"] = {"value": year, "coords": [x, y, w, h]}
            logger.info(f"Année d'étude extraite: {year} avec coordonnées: {x}, {y}, {w}, {h}")
        except ValueError:
            student["year"] = {"value": year, "coords": None}
            logger.warning(f"L'année {year} n'a pas pu être localisée dans l'image.")

    if student.get("ine") is None and ine_match:
        ine_number = ine_match.group(1)
        ine_letter = ine_match.group(2)
        ine = f"{ine_number} {ine_letter}"

        try:
            ine_number_word_index = text_boxes["text"].index(ine_number)
            x, y, w, h = (
                text_boxes["left"][ine_number_word_index],
                text_boxes["top"][ine_number_word_index],
                text_boxes["width"][ine_number_word_index],
                text_boxes["height"][ine_number_word_index],
            )

            ine_letter_word_index = text_boxes["text"].index(ine_letter)
            dx, _, dw, dh = (
                text_boxes["left"][ine_letter_word_index],
                text_boxes["top"][ine_letter_word_index],
                text_boxes["width"][ine_letter_word_index],
                text_boxes["height"][ine_letter_word_index],
            )

            student["ine"] = {"value": ine, "coords": [x, y, dx + dw - x, h]}
            logger.info(f"INE extrait: {ine} avec coordonnées: {x}, {y}, {dx + dw - x}, {h}")
        except ValueError:
            try:
                word_index = text_boxes["text"].index(f"{ine_number}{ine_letter}")
                x, y, w, h = (
                    text_boxes["left"][word_index],
                    text_boxes["top"][word_index],
                    text_boxes["width"][word_index],
                    text_boxes["height"][word_index],
                )
                student["ine"] = {"value": ine, "coords": [x, y, w, h]}
                logger.info(f"INE extrait: {ine} avec coordonnées: {x}, {y}, {w}, {h}")
            except ValueError:
                student["ine"] = {"value": ine, "coords": None}
                logger.warning(f"L'INE {ine} n'a pas pu être localisé dans l'image.")

    if student.get("student_num") is None and student_num_match:
        student_num = student_num_match.group()
        try:
            word_index = text_boxes["text"].index(student_num)
            x, y, w, h = (
                text_boxes["left"][word_index],
                text_boxes["top"][word_index],
                text_boxes["width"][word_index],
                text_boxes["height"][word_index],
            )
            student["student_num"] = {"value": student_num, "coords": [x, y, w, h]}
            logger.info(f"Numéro étudiant extrait: {student_num} avec coordonnées: {x}, {y}, {w}, {h}")
        except ValueError:
            student["student_num"] = {"value": student_num, "coords": None}
            logger.warning(f"Le numéro étudiant {student_num} n'a pas pu être localisé dans l'image.")

    return student

def preprocess_image(params, image_path):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params["kernel_size"])
    image = cv2.imread(image_path)

    if params["bilateralFilter"]:
        logger.info(f"Application du filtre bilatéral... [{params['bilateralFilter'][0]}, {params['bilateralFilter'][1]}, {params['bilateralFilter'][2]}]")
        image = cv2.bilateralFilter(image, params["bilateralFilter"][0], params["bilateralFilter"][1], params["bilateralFilter"][2])

    if params["blur"]:
        logger.info(f"Application du flou... [{params['ksize']}]")
        image = cv2.blur(image, (params["ksize"], params["ksize"])) if isinstance(params["ksize"], int) else cv2.blur(image, params["ksize"])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if params["equal_hist"]:
        logger.info("Application de l'égalisation de l'histogramme...")
        image = cv2.equalizeHist(image)

    if params["sharpen"]:
        logger.info("Application du filtre de netteté...")
        image = unsharp_mask(image)

    binary_count = 0

    student = {
        "fname": None,
        "lname": None,
        "year": None,
        "ine": None,
        "student_num": None,
    }

    while binary_count < 3 and not(check_keys_found(student)):
        _, binary_image = cv2.threshold(image, params["binary"] + binary_count * 5, 255, cv2.THRESH_BINARY)
        logger.info(f"Binarisation de l'image... ({params['binary'] + binary_count * 5})")
        student = extract_student_card(binary_image, student)

        command_count = 0

        while command_count < 3 and not(check_keys_found(student)):
            logger.info("Application des opérations d'ouverture et de fermeture...")
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            student = extract_student_card(binary_image, student)
            command_count += 1
        
        binary_count += 1

    return student

def show_student_card(image_path):
    logger.info(f"Chargement de l'image à partir de : {image_path}")
    image = cv2.imread(image_path)
    characteristics = extract_image_characteristics(image_path)
    params = extract_params(characteristics)

    logger.info("Prétraitement de l'image pour extraire les caractéristiques de l'étudiant...")
    student = preprocess_image(params, image_path)

    logger.info("Recherche des visages dans l'image...")
    faces = find_faces(image_path)

    for (x, y, w, h) in faces:
        logger.info(f"Visage trouvé à la position : ({x}, {y}), taille : ({w}, {h})")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(image, 'face', (x - 50, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, 2)

    for key, value in student.items():
        if value:
            if value['coords']:
                coords = value['coords']
                logger.info(f"Coordonnées de {key} trouvées : {coords}")
                cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 4)
                cv2.putText(image, key, (coords[0] - 100, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, 2)

    logger.info(f"Affichage de l'image annotée : {image_path}")
    cv2.imshow(image_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_classification(image_path):  
    characteristics = extract_image_characteristics(image_path)

    image = cv2.imread(image_path)
    image = cv2.bilateralFilter(image, 10, 75, 75)

    width, _ = characteristics['resolution']

    if width >= 1920:
        image = cv2.blur(image, (3, 3))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = unsharp_mask(image)

    if characteristics['brightness'] >= 200:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    elif characteristics['brightness'] >= 150:
        _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    else:
        _, image = cv2.threshold(image, 65, 255, cv2.THRESH_BINARY)

    return image

patterns = {
    "carte_fidelite": r"\b(fidélité|ikea|franprix|maison|ma carte|carte de)\b",
    "carte_etudiant": r"\b(étudiant|université|crous|ine|etudiant)\b",
    "carte_identite": r"\b(république française|carte nationale|identité|nationalité|nom|prénom|sexe)\b"
}

def extract_features(text, patterns):
    extracted = []
    for _, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        extracted.extend(matches)
    if extracted == []:
        return "MA CARTE DE FIDÉLITÉ IKEA FRANPRIX MAISON"
    else:
        return " ".join(extracted)

def extract_text(image):
    text = pytesseract.image_to_string(image, lang='fra')
    return extract_features(text.strip(), patterns)

def create_classifier(train_data):
    labels, texts = zip(*train_data)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', SVC(kernel='linear', probability=True))
    ])
    pipeline.fit(texts, labels)
    return pipeline

training_data = [
    ("carte_etudiant", "ETUDIANT INE étudiant UNIVERSITE CROUS"),
    ("carte_identite", "RÉPUBLIQUE FRANÇAISE CARTE NATIONALE IDENTITÉ Nationalité Française Nom Prénom Signature Sexe"),
    ("carte_fidelite", "MA CARTE DE FIDÉLITÉ IKEA FRANPRIX MAISON")
]

classifier = create_classifier(training_data)

def classify_card(text):
    prediction = classifier.predict([text])
    return prediction[0]

def extract_type_from_path(file_path):
    match = re.search(r"ressources_projet/([a-zA-Z_]+)\d+\.\w+", file_path)
    if match:
        return match.group(1)
    return None

def compare_card_type(file_path, card_type):
    extracted_type = extract_type_from_path(file_path)
    if not extracted_type:
        logger.error(f"Could not extract type from path: {file_path}")
        return False

    normalized_card_type = card_type.replace("_", "")
    return extracted_type == normalized_card_type

def process_image(image_path):
    print(f"\n--- Traitement de l'image : {image_path} ---")
    processed_image = preprocess_classification(image_path)
    text = extract_text(processed_image)
    card_type = classify_card(text)
    if compare_card_type(image_path, card_type):
        print(f"Type de carte : {card_type} ✅")
    else: 
        print(f"Type de carte : {card_type} ❌")
    if card_type == 'carte_etudiant':
        show_student_card(image_path)
    elif card_type == 'carte_identite':
        extract_id(image_path)