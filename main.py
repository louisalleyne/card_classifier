from utils import process_image
import os

if __name__ == "__main__":
    
    directory = "ressources_projet"

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            if os.path.isfile(image_path):
              process_image(image_path)
    else:
        print(f"Le répertoire {directory} n'existe pas.")