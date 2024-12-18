import os
import sys
from cv2 import imread
from utils import process_image

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Erreur: Fichier '{image_path}' n'existe pas.")
        sys.exit(1)
    
    try:
        process_image(image_path)
    except (IOError, SyntaxError) as e:
        print(f"Erreur: '{image_path}' n'est pas un format d'image valide.")
        sys.exit(1)

if __name__ == "__main__":
    main()
