# image_functions.py
import cv2


def rgb_to_lab(image_rgb):
    """
    Convertit une image du format RGB en Lab.

    :param image_rgb: Image en format RGB.
    :return: Image convertie en format Lab.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)


def lab_to_rgb(image_lab):
    """
    Convertit une image du format Lab en RGB.

    :param image_lab: Image en format Lab.
    :return: Image convertie en format RGB.
    """
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)


def apply_bilateral_filter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Applique un filtre bilatéral à l'image RGB donnée pour réduire le bruit.

    :param image_rgb: Image en format RGB.
    :param d: Diamètre de chaque pixel voisin.
    :param sigmaColor: Filtre sigma dans l'espace de couleur.
    :param sigmaSpace: Filtre sigma dans l'espace de coordonnées.
    :return: Image RGB filtrée.
    """
    return cv2.bilateralFilter(image_rgb, d, sigmaColor, sigmaSpace)
