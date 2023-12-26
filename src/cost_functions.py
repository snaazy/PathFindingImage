import numpy as np


def cost_function_lab(point1, point2, image_lab):
    """
    Calcule la distance de couleur entre deux points dans l'espace de couleur Lab.
    Utilise les composantes L, a, et b pour calculer la distance euclidienne.

    :param point1: Tuple des coordonnées du premier point.
    :param point2: Tuple des coordonnées du second point.
    :param image_lab: Image dans l'espace de couleur Lab.
    :return: Distance euclidienne entre les deux points dans l'espace Lab.
    """
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def cost_function_labDif(point1, point2, image_lab):
    """
    Calcule la différence de luminosité (composante L) entre deux points dans l'espace Lab,
    en ignorant les composantes de couleur a et b.

    :param point1: Tuple des coordonnées du premier point.
    :param point2: Tuple des coordonnées du second point.
    :param image_lab: Image dans l'espace de couleur Lab.
    :return: Différence absolue des composantes L entre les deux points.
    """
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return abs(L1 - L2)


def cost_function_intensity(point1, point2, image_lab):
    intensity1 = image_lab[point1[0], point1[1], 0] / 255.0
    intensity2 = image_lab[point2[0], point2[1], 0] / 255.0
    return abs(intensity1 - intensity2)


def cost_function_local_contrast(point1, point2, image_lab, window_size=5):
    height, width = image_lab.shape[:2]
    y1, x1 = point1
    y2, x2 = point2

    half_window = window_size // 2
    x1_min = max(0, x1 - half_window)
    x1_max = min(width - 1, x1 + half_window)
    y1_min = max(0, y1 - half_window)
    y1_max = min(height - 1, y1 + half_window)

    x2_min = max(0, x2 - half_window)
    x2_max = min(width - 1, x2 + half_window)
    y2_min = max(0, y2 - half_window)
    y2_max = min(height - 1, y2 + half_window)

    # calcule la moyenne de la luminance autour de deux points
    window1 = image_lab[y1_min : y1_max + 1, x1_min : x1_max + 1, 0]
    window2 = image_lab[y2_min : y2_max + 1, x2_min : x2_max + 1, 0]

    mean_intensity1 = np.mean(window1)
    mean_intensity2 = np.mean(window2)

    # calcule le contraste local en utilisant la différence des moyennes de luminance
    local_contrast = abs(mean_intensity1 - mean_intensity2)

    return local_contrast
