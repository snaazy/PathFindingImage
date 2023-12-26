# main.py
import tkinter as tk
from tkinter import Button, Label, Radiobutton, IntVar, filedialog
import heapq
import time
from PIL import Image, ImageTk
import cv2
import numpy as np
from cost_functions import *
from image_processing import *
from collections import defaultdict
from image_processing import *


# Variables globales
points = []
image = None
window = None
canvas = None
instruction_label = None
imgtk = None
graph = {}  # Le graphe représenté par un dictionnaire
points_raw = []
original_image = None
delay_ms = 1  # ajustez ici le temps en ms pour le dessin progressif du chemin
cost_function_choice = None


# Fonctions Dijkstra et autres fonctions principales ici
# Algorithme de Dijkstra pour le calcul du plus court chemin
def dijkstra(image_lab, start, end, cost_function):
    height, width = image_lab.shape[:2]
    visited = np.full((height, width), False, dtype=bool)
    distance_map = np.full((height, width), np.inf)
    parent_map = np.full((height, width, 2), -1, dtype=int)

    distance_map[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        dist, current_node = heapq.heappop(priority_queue)
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
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    path = [end]
    while path[-1] != start:
        parent = tuple(parent_map[path[-1]])
        path.append(parent)

    return path[::-1]


def update_instructions(text):
    instruction_label.config(text=text)


def chemin_couleur_intensite(image_lab, path):
    """
    Trace le chemin le plus court entre deux points sur une image et colorie chaque segment
    en fonction de l'intensité moyenne des pixels le long du segment.

    :param image_lab: Image dans l'espace de couleur Lab.
    :param path: Liste des points (y, x) constituant le chemin le plus court.
    :return: Image avec le chemin tracé et colorié en fonction de l'intensité.

    Cette fonction copie d'abord l'image originale. Pour chaque segment du chemin, elle calcule
    l'intensité moyenne entre deux points consécutifs. Cette intensité est ensuite utilisée pour
    déterminer la couleur du segment grâce à la fonction `determiner_couleur_intensite`.
    Chaque segment est dessiné sur l'image avec la couleur correspondante à son intensité.
    """
    colored_path_image = original_image.copy()
    for i in range(len(path) - 1):
        point1 = path[i]
        point2 = path[i + 1]
        avg_intensity = intensite_moyenne(image_lab, point1, point2)
        color = determiner_couleur_intensite(avg_intensity)
        cv2.line(colored_path_image, point1[::-1], point2[::-1], color, 2)
    return colored_path_image


def determiner_couleur_intensite(intensity):
    """
    Détermine la couleur d'un segment de chemin en fonction de son intensité.

    :param intensity: Valeur d'intensité du segment (valeur entière de 0 à 255).
    :return: Tuple représentant la couleur (R, G, B) correspondante à l'intensité.

    La fonction normalise d'abord l'intensité sur une échelle de 0 à 1. Ensuite, en fonction
    de cette valeur normalisée, elle détermine la couleur du segment en interpolant entre
    différentes couleurs : du bleu au cyan, du cyan au vert, du vert au jaune, et du jaune
    au rouge. Cette interpolation crée un effet de gradient, permettant une visualisation
    plus détaillée et graduelle des changements d'intensité le long du chemin.
    """
    intensite_normalise = intensity / 255.0

    if intensite_normalise < 0.25:
        # Bleu à cyan
        return interpoler_couleurs(
            (0, 0, 255), (0, 255, 255), intensite_normalise / 0.25
        )
    elif intensite_normalise < 0.5:
        # Cyan à vert
        return interpoler_couleurs(
            (0, 255, 255), (0, 255, 0), (intensite_normalise - 0.25) / 0.25
        )
    elif intensite_normalise < 0.75:
        # Vert à jaune
        return interpoler_couleurs(
            (0, 255, 0), (255, 255, 0), (intensite_normalise - 0.5) / 0.25
        )
    else:
        # Jaune à rouge
        return interpoler_couleurs(
            (255, 255, 0), (255, 0, 0), (intensite_normalise - 0.75) / 0.25
        )


def interpoler_couleurs(color_start, color_end, factor):
    """
    Interpole entre deux couleurs RGB selon un facteur donné.

    :param color_start: Tuple RGB de la couleur de départ (par exemple, (255, 0, 0) pour le rouge).
    :param color_end: Tuple RGB de la couleur de fin (par exemple, (0, 255, 0) pour le vert).
    :param factor: Un facteur de 0 à 1 indiquant le degré d'interpolation.
    :return: Tuple RGB représentant la couleur interpolée.

    """
    return tuple(
        int(start_val + (end_val - start_val) * factor)
        for start_val, end_val in zip(color_start, color_end)
    )


def intensite_moyenne(image_lab, point1, point2):
    """
    Calcule l'intensité moyenne des pixels le long d'une ligne entre deux points sur une image.

    :param image_lab: Image dans l'espace de couleur Lab.
    :param point1: Tuple (y, x) représentant le premier point de la ligne.
    :param point2: Tuple (y, x) représentant le second point de la ligne.
    :return: La valeur moyenne d'intensité des pixels le long de la ligne.

    Cette fonction dessine d'abord une ligne entre `point1` et `point2` sur une image noire
    de la même taille que `image_lab`. Elle identifie ensuite les pixels où la ligne a été
    dessinée. Si aucun pixel n'est trouvé sur la ligne (ligne vide), elle retourne 0.
    Sinon, elle calcule et retourne l'intensité moyenne des pixels le long de cette ligne
    en utilisant la composante de luminance (L) de l'espace de couleur Lab.
    """
    line = cv2.line(np.zeros(image_lab.shape[:2]), point1[::-1], point2[::-1], 1, 1)
    indices = np.where(line == 1)
    if len(indices[0]) == 0:  # Aucun pixel sur la ligne
        return 0
    intensities = image_lab[indices[0], indices[1], 0]
    return np.mean(intensities)


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
    original_image = original_image.copy()  # reinitialiser l'image originale
    clear_path()
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


def selection_fonction_cout():
    global window, cost_function_choice
    cost_function_choice = IntVar()
    frame = tk.Frame(window)
    frame.pack(side="top")

    Label(frame, text="Choisissez une fonction de coût :").pack()
    Radiobutton(frame, text="Lab", variable=cost_function_choice, value=1).pack(
        anchor=tk.W
    )
    Radiobutton(frame, text="LabDif", variable=cost_function_choice, value=2).pack(
        anchor=tk.W
    )
    Radiobutton(frame, text="Intensité", variable=cost_function_choice, value=3).pack(
        anchor=tk.W
    )
    Radiobutton(
        frame,
        text="Contraste Local (peut être long)",
        variable=cost_function_choice,
        value=4,
    ).pack(anchor=tk.W)


def show_reset_button():
    reset_btn = Button(window, text="Réinitialiser", command=reset_selection)
    reset_btn.pack(side="bottom")


def create_graph(image_lab):
    """
    Crée un graphe représentant l'image donnée, où chaque pixel est un sommet.

    Cette fonction crée un graphe basé sur les pixels de l'image fournie. Chaque pixel est
    traité comme un sommet du graphe. Les voisins de chaque pixel (sommet) sont ajoutés en
    considérant une connectivité de 4 (haut, bas, gauche, droite).

    :param image_lab: L'image sur laquelle le graphe est basé, généralement convertie en
                      espace de couleur Lab* pour les calculs de coût.
    :global original_image: L'image originale, utilisée ici pour référence.

    :return: Un dictionnaire représentant le graphe. Les clés sont des tuples (y, x)
             représentant les coordonnées des sommets (pixels), et les valeurs sont des
             listes de tuples représentant les voisins de chaque sommet.
    """
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


def show_button():
    btn = Button(window, text="Trouver le chemin le plus court", command=trouver_pcc)
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


def trouver_pcc():
    """
    Trouve et affiche le plus court chemin (PCC) entre deux points sélectionnés sur l'image.

    Cette fonction utilise l'algorithme de Dijkstra pour calculer le plus court chemin entre
    deux points sélectionnés, en fonction de la fonction de coût choisie. Le chemin est ensuite
    dessiné sur l'image, avec les couleurs variant selon l'intensité moyenne des pixels sur
    le chemin.

    :global image: L'image sur laquelle le PCC est calculé.
    :global points: Liste des points sélectionnés (départ et arrivée).
    :global canvas: Le canvas de Tkinter utilisé pour l'affichage.
    :global imgtk: ImageTk utilisée pour l'affichage sur le canvas.

    La fonction vérifie d'abord s'il y a exactement deux points sélectionnés. Si oui, elle
    détermine la fonction de coût en fonction du choix de l'utilisateur. L'algorithme de Dijkstra
    est ensuite exécuté pour trouver le chemin. Chaque segment du chemin est coloré en fonction
    de l'intensité moyenne des pixels entre les deux points du segment. Le chemin est dessiné
    progressivement, avec un délai entre chaque segment pour une visualisation dynamique.

    Les instructions et les coordonnées du chemin sont mises à jour en conséquence.
    """
    global image, points, canvas, imgtk
    if len(points) == 2:
        start, end = points
        # selection de la fonction de cout
        if cost_function_choice.get() == 1:
            cost_function = cost_function_lab
        elif cost_function_choice.get() == 2:
            cost_function = cost_function_labDif
        elif cost_function_choice.get() == 3:
            cost_function = cost_function_intensity
        elif cost_function_choice.get() == 4:
            cost_function = cost_function_local_contrast
        else:
            print("Veuillez sélectionner une fonction de coût")
            return

        path = dijkstra(image, start[::-1], end[::-1], cost_function)

        start_filtered = apply_coordinate_transform(start[0], start[1])
        end_filtered = apply_coordinate_transform(end[0], end[1])

        # dessine le chemin progressivement
        for i in range(len(path) - 1):
            point1 = path[i]
            point2 = path[i + 1]
            avg_intensity = intensite_moyenne(image, point1, point2)
            color = determiner_couleur_intensite(avg_intensity)
            cv2.line(original_image, point1[::-1], point2[::-1], color, 2)
            refresh_image()
            canvas.update()
            window.after(delay_ms)

        update_instructions(
            "Chemin trouvé. Utilisez le bouton Réinitialiser pour recommencer. (vous pouvez voir les coordonnées du PCC dans le terminal !)"
        )

        formatted_path = "\n".join(
            [
                f"Sommet {i+1} de coordonnées: ({point[0]}, {point[1]})"
                for i, point in enumerate(path)
            ]
        )
        print(
            "*** PLUS COURT CHEMIN ***\n",
            formatted_path,
            "\n *** FIN DES COORDONNEES DU PLUS COURT CHEMIN ***",
        )


def print_graph(graph):
    print("Graphe :")
    for node, neighbors in graph.items():
        print(f"Sommets {node} a pour voisins : {neighbors}")


def main():
    global image, window, canvas, instruction_label, imgtk, graph
    window = tk.Tk()
    window.title("Trouver le chemin le plus court")

    instruction_label = Label(window, text="Veuillez sélectionner le point 1.")
    instruction_label.pack(side="top")

    canvas = tk.Canvas(window, width=600, height=400)
    canvas.pack(side="top", fill="both", expand=True)

    selection_fonction_cout()

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
