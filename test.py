import cv2
import numpy as np

# Chargement de l'image
nom_fichier_image = 'Mona_LisaColor.png'  # Remplacez 'votre_image.jpg' par le chemin de votre propre image
image = cv2.imread(nom_fichier_image)

if image is None:
    print("L'image n'a pas pu être chargée.")
else:
    # Appliquer le filtre bilateral pour lisser l'image
    image_lissee = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Créer une nouvelle image côte à côte
    image_combinee = np.hstack((image, image_lissee))

    # Afficher les deux images côte à côte dans une fenêtre graphique
    cv2.imshow('Image Originale vs. Image Lissée', image_combinee)

    # Attendre une touche et fermer la fenêtre lorsque n'importe quelle touche est enfoncée
    cv2.waitKey(0)
    cv2.destroyAllWindows()
