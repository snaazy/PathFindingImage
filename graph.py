import heapq
from queue import PriorityQueue
from tkinter import filedialog
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import networkx as nx
from collections import defaultdict

# variables globales
points = []
image = None
window = None
canvas = None
instruction_label = None
imgtk = None  
graph = {}  # Le graphe représenté par un dictionnaire
points_raw = []
original_image = None


#  -----  fonctions de calcul de coût -----

""" Les fonctions de calculs de coûts dépendent du type d'image qu'on traite et des informations qu'on dispose sur cette image."""

""" tient compte de la difference de couleur et de luminosite entre les pixels."""


def cost_function_lab(point1, point2, image_lab):
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


""" tient compte uniquement de la difference de luminosité (L) tout en ignorant la chrominance.
-> cette fonction serait plus intéressante pour Mona Lisa car bcp de bruit ? """


def cost_function_labDif(point1, point2, image_lab):
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return abs(L1 - L2)


""" normalise les valeurs de luminance (L) et calcule la différence d'intensité lumineuse.
-> cette fonction serait plus intéressante pour Mona Lisa car bcp de bruit ?"""


def cost_function_intensity(point1, point2, image_lab):
    intensity1 = image_lab[point1[0], point1[1], 0] / 255.0
    intensity2 = image_lab[point2[0], point2[1], 0] / 255.0
    return abs(intensity1 - intensity2)


""" peut etre long a s'executer sur des images de grandes résolutions, plus la fenetre est grande, plus le calcul
devient intensif.  """


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


# Fonction de conversion RGB en Lab*
def rgb_to_lab(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)


# Fonction de conversion Lab* en RGB
def lab_to_rgb(image_lab):
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)


# Fonction d'application du filtre gaussien
def apply_bilateral_filter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image_rgb, d, sigmaColor, sigmaSpace)


# ---------------------------------------------------------------------------

# Algorithme de Dijkstra pour le calcul du plus court chemin
def dijkstra(image_lab, start, end, cost_function):
    height, width = image_lab.shape[:2]
    visited = np.full((height, width), False, dtype=bool)
    distance_map = np.full((height, width), np.inf)
    parent_map = np.full((height, width, 2), -1, dtype=int)

    distance_map[start] = 0
    priority_queue = PriorityQueue()
    priority_queue.put((0, start))

    while not priority_queue.empty():
        dist, current_node = priority_queue.get()
        if visited[current_node]:
            continue
        visited[current_node] = True

        if current_node == end:
            break

        for dx, dy in [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]:  # 8-connexité
            new_x = current_node[1] + dx
            new_y = current_node[0] + dy
            neighbor = (new_y, new_x)

            if 0 <= new_x < width and 0 <= new_y < height and not visited[neighbor]:
                cost = cost_function(current_node, neighbor, image_lab)
                new_dist = dist + cost

                if new_dist < distance_map[neighbor]:
                    distance_map[neighbor] = new_dist
                    parent_map[neighbor] = current_node
                    priority_queue.put((new_dist, neighbor))

    path = [end]
    while path[-1] != start:
        parent = tuple(parent_map[path[-1]])
        path.append(parent)

    return path[::-1]


def update_instructions(text):
    instruction_label.config(text=text)


# Recherche du chemin le plus court
def find_shortest_path():
    global image, points, canvas, imgtk
    if len(points) == 2:
        start, end = points

        start_filtered = apply_coordinate_transform(start[0], start[1])
        end_filtered = apply_coordinate_transform(end[0], end[1])

        """ CHANGEZ ICI DANS L'APPEL DE LA FONCTION DIJKSTRA, LA FONCTION DE COUT,
         cost_function_local_contrast PEUT ETRE TRES LONGUE A S'EXECUTER ! """
        path = dijkstra(
            image,
            start_filtered[::-1],
            end_filtered[::-1],
            cost_function_local_contrast,
        )

        # dessine le chemin sur l'image
        for i in range(len(path) - 1):
            cv2.line(original_image, path[i][::-1], path[i + 1][::-1], (0, 255, 0), 2)

        refresh_image()

        update_instructions(
            "Chemin trouvé. Utilisez le bouton Réinitialiser pour recommencer. (vous pouvez voir les coordonnées du PCC dans le terminal !)"
        )
        formatted_path = "\n".join(
            [f"Point {i+1}: ({point[0]}, {point[1]})" for i, point in enumerate(path)]
        )
        print("Chemin le plus court :\n", formatted_path)


def show_button():
    btn = Button(
        window, text="Trouver le chemin le plus court", command=find_shortest_path
    )
    btn.pack(side="bottom")


def on_canvas_click(event):
    global points, points_raw, image, canvas, imgtk
    if len(points) < 2:
        x, y = event.x, event.y
        points_raw.append((x, y))
        x_filtered, y_filtered = apply_coordinate_transform(x, y)
        points.append((x_filtered, y_filtered))
        cv2.circle(original_image, (x_filtered, y_filtered), 5, (0, 255, 0), -1)
        refresh_image()
        if len(points) == 1:
            update_instructions(
                f"Point {len(points)} sélectionné. Sélectionnez un autre point."
            )
        elif len(points) == 2:
            update_instructions(
                f"Point 2 sélectionné : ({x}, {y}). Cliquez sur le bouton pour calculer le chemin."
            )
            show_button()


def apply_coordinate_transform(x, y):
    return x, y


def refresh_image():
    global canvas, original_image, window, imgtk
    if original_image is not None:
        im = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=im)
        canvas.create_image(0, 0, anchor="nw", image=imgtk)


def reset_points():
    global points, original_image, graph
    points = []
    original_image = original_image.copy()  # Réinitialisez l'image originale
    clear_path()  # Ajoutez cette ligne pour effacer le chemin
    refresh_image()
    update_instructions("Veuillez sélectionner le point 1.")


def clear_path():
    global original_image
    original_image = original_image.copy()


def reset_selection():
    global image, original_image, points, canvas, imgtk
    points = []

    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        if original_image is None:
            print("Erreur : Impossible de charger l'image.")
            return

        # applique le filtre bilatéral et convertir en Lab*
        image_rgb_blurred = apply_bilateral_filter(original_image)
        image = rgb_to_lab(image_rgb_blurred)

        refresh_image()
        update_instructions("Veuillez sélectionner le point 1.")


def show_reset_button():
    reset_btn = Button(window, text="Réinitialiser", command=reset_selection)
    reset_btn.pack(side="bottom")


# Fonction pour charger une image
def load_image():
    global image, canvas, window, imgtk

    file_path = filedialog.askopenfilename()
    if file_path:
        # Charger l'image
        original_image = cv2.imread(file_path)
        if original_image is None:
            print("Erreur : Impossible de charger l'image.")
            return

        # applique le filtre gaussien sur l'image RGB
        image_rgb_blurred = apply_bilateral_filter(original_image)

        # convertit l'image lissée en Lab*
        image = rgb_to_lab(image_rgb_blurred)

        refresh_image()

        # redimensionner la fenêtre en fonction de la taille de l'image
        window.geometry(f"{image.shape[1]}x{image.shape[0]}")

        update_instructions("Veuillez sélectionner le point 1.")


def print_graph(graph):
    print("Graphe :")
    for node, neighbors in graph.items():
        print(f"Sommets {node} a pour voisins : {neighbors}")


def create_graph(image_lab):
    global original_image
    graph = {}
    height, width = image_lab.shape[:2]

    for y in range(height):
        for x in range(width):
            neighbors = []
            if x > 0:
                neighbors.append((y, x - 1))
            if y > 0:
                neighbors.append((y - 1, x))
            graph[(y, x)] = neighbors

            """ pour afficher chaque sommet en vert, logiquement ca devrait afficher une image remplie 
             en vert car chaque pixel = un sommet  """
            # cv2.circle(original_image, (x, y), 1, (0, 255, 0), -1)

    return graph


def main():
    global image, window, canvas, instruction_label, imgtk, graph
    window = tk.Tk()
    window.title("Trouver le chemin le plus court")

    instruction_label = Label(window, text="Veuillez sélectionner le point 1.")
    instruction_label.pack(side="top")

    canvas = tk.Canvas(window, width=600, height=400)
    canvas.pack(side="top", fill="both", expand=True)

    reset_selection()

    # création du graphe
    graph = create_graph(image)

    # print_graph(graph)  # Imprimez le graphe

    reset_btn = Button(window, text="Choisir une autre image", command=reset_selection)
    reset_btn.pack(side="bottom")

    reset_button = Button(
        window,
        text="Recommencer sur cette image (conserve l'ancien chemin)",
        command=reset_points,
    )
    reset_button.pack(side="bottom")
    canvas.bind("<Button-1>", on_canvas_click)

    window.mainloop()


if __name__ == "__main__":
    main()
