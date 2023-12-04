import heapq
from tkinter import filedialog
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

# Variables globales
points = []
image = None
window = None
canvas = None
instruction_label = None
imgtk = None  # Pour stocker l'image Tkinter
graph = {}  # Le graphe représenté par un dictionnaire

# Exemple de fonctions de calcul de coût
def cost_function_lab(point1, point2, image_lab):
    """
    Calcule le coût entre deux points dans l'espace Lab*.

    Args:
        point1 (tuple): Coordonnées du premier point (y, x).
        point2 (tuple): Coordonnées du deuxième point (y, x).
        image_lab (numpy.ndarray): Image dans l'espace Lab*.

    Returns:
        float: Coût entre les deux points.
    """
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def cost_function_labDif(point1, point2, image_lab):
    """
    Calcule le coût de différence de luminosité entre deux points dans l'espace Lab*.

    Args:
        point1 (tuple): Coordonnées du premier point (y, x).
        point2 (tuple): Coordonnées du deuxième point (y, x).
        image_lab (numpy.ndarray): Image dans l'espace Lab*.

    Returns:
        float: Coût de différence de luminosité entre les deux points.
    """
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return abs(L1 - L2)

# Fonction de conversion RGB en Lab*
def rgb_to_lab(image_rgb):
    """
    Convertit une image RGB en Lab*.

    Args:
        image_rgb (numpy.ndarray): Image RGB.

    Returns:
        numpy.ndarray: Image dans l'espace Lab*.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)

# Fonction de conversion Lab* en RGB
def lab_to_rgb(image_lab):
    """
    Convertit une image Lab* en RGB.

    Args:
        image_lab (numpy.ndarray): Image dans l'espace Lab*.

    Returns:
        numpy.ndarray: Image RGB.
    """
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)

# Fonction d'application du filtre gaussien
def apply_bilateral_filter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Applique un filtre bilatéral sur une image RGB.

    Args:
        image_rgb (numpy.ndarray): Image RGB.
        d (int): Taille de la fenêtre du filtre.
        sigmaColor (float): Écart-type en couleur.
        sigmaSpace (float): Écart-type en espace.

    Returns:
        numpy.ndarray: Image filtrée.
    """
    return cv2.bilateralFilter(image_rgb, d, sigmaColor, sigmaSpace)

# Fonction pour réinitialiser la sélection de points
def reset_selection():
    """
    Réinitialise la sélection de points et charge l'image originale filtrée en Lab*.
    """
    global image, points, canvas, imgtk
    points = []  # Réinitialiser les points sélectionnés
    
    # Charger l'image originale
    original_image = cv2.imread('scanner.png')
    if original_image is None:
        print("Erreur : Impossible de charger l'image.")
        return
    
    # Appliquer le filtre gaussien sur l'image RGB
    image_rgb_blurred = apply_bilateral_filter(original_image)
    
    # Convertir l'image lissée en Lab*
    image = rgb_to_lab(image_rgb_blurred)
    
    refresh_image()
    update_instructions("Veuillez sélectionner le point 1.")

# Fonction pour charger une image
def load_image():
    """
    Charge une image depuis un fichier et la filtre en Lab*.
    """
    global image, canvas, window, imgtk

    file_path = filedialog.askopenfilename()
    if file_path:
        # Charger l'image
        original_image = cv2.imread(file_path)
        if original_image is None:
            print("Erreur : Impossible de charger l'image.")
            return

        # Appliquer le filtre gaussien sur l'image RGB
        image_rgb_blurred = apply_bilateral_filter(original_image)

        # Convertir l'image lissée en Lab*
        image = rgb_to_lab(image_rgb_blurred)

        # Mettre à jour l'image affichée dans Tkinter
        refresh_image()

        # Redimensionner la fenêtre en fonction de la taille de l'image
        window.geometry(f"{image.shape[1]}x{image.shape[0]}")

        update_instructions("Veuillez sélectionner le point 1.")

# Fonction pour créer le graphe
def create_graph(image_lab):
    """
    Crée un graphe représentant les connexions entre les pixels voisins.

    Args:
        image_lab (numpy.ndarray): Image dans l'espace Lab*.

    Returns:
        dict: Graphe représenté par un dictionnaire.
    """
    graph = {}
    height, width = image_lab.shape[:2]

    for y in range(height):
        for x in range(width):
            neighbors = []
            if x > 0:
                neighbors.append((y, x - 1))  # Connexions horizontales
            if y > 0:
                neighbors.append((y - 1, x))  # Connexions verticales
            graph[(y, x)] = neighbors

    return graph

# Fonction principale pour trouver le chemin le plus court
def find_shortest_path():
    """
    Trouve le chemin le plus court entre deux points sélectionnés.
    """
    global image, points, canvas, imgtk, graph
    if len(points) == 2:
        start, end = points
        path = dijkstra(image, start[::-1], end[::-1], cost_function_lab)

        # Dessiner le chemin sur l'image
        for i in range(len(path) - 1):
            cv2.line(image, path[i][::-1], path[i+1][::-1], (0, 255, 0), 2)

        # Mettre à jour l'image affichée dans Tkinter
        refresh_image()

        # Le chemin trouvé est simplement path
        print("Chemin le plus court :", path)

# Fonction pour mettre à jour les instructions
def update_instructions(text):
    """
    Met à jour les instructions affichées dans l'interface utilisateur.

    Args:
        text (str): Nouveau texte d'instruction.
    """
    instruction_label.config(text=text)

# Fonction pour dessiner les sommets du graphe
def draw_graph_vertices(image, graph):
    """
    Dessine les sommets du graphe sur l'image.

    Args:
        image (numpy.ndarray): Image sur laquelle dessiner les sommets.
        graph (dict): Graphe représenté par un dictionnaire.
    """
    for vertex in graph.keys():
        y, x = vertex  # Les coordonnées du sommet
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Dessine un petit cercle rouge à la position du sommet

# Fonction pour actualiser l'image affichée
def refresh_image():
    """
    Actualise l'image affichée dans l'interface utilisateur.
    """
    global canvas, image, window, imgtk
    image_for_tk = lab_to_rgb(image)
    im = Image.fromarray(image_for_tk)
    imgtk = ImageTk.PhotoImage(image=im)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

# Fonction principale
def main():
    global image, window, canvas, instruction_label, imgtk, graph
    window = tk.Tk()
    window.title("Trouver le chemin le plus court")

    instruction_label = Label(window, text="Veuillez sélectionner le point 1.")
    instruction_label.pack(side="top")

    # Créer le canvas ici pour éviter l'erreur AttributeError
    canvas = tk.Canvas(window, width=600, height=400)  # Assurez-vous que la taille est correcte pour votre image
    canvas.pack(side="top", fill="both", expand=True)
    
    # Initialisation de l'image avec le filtre gaussien et conversion en Lab*
    reset_selection()  # Cette fonction va maintenant charger l'image, appliquer le filtre gaussien, convertir en Lab* et rafraîchir l'image

    # Créez le graphe en utilisant la fonction create_graph
    graph = create_graph(image)

    # Dessine les sommets du graphe
    draw_graph_vertices(image, graph)

    # Créer un bouton pour réinitialiser la sélection de points
    reset_btn = Button(window, text="Réinitialiser", command=reset_selection)
    reset_btn.pack(side="bottom")

    canvas.bind("<Button-1>", on_canvas_click)  # S'assurer que le canvas est prêt avant de lier les événements
  
    window.mainloop()

if __name__ == "__main__":
    main()
