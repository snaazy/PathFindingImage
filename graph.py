import heapq
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
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def cost_function_labDif(point1, point2, image_lab):
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return abs(L1 - L2)

# Fonction de conversion RGB en Lab*
def rgb_to_lab(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)

# Fonction de conversion Lab* en RGB
def lab_to_rgb(image_lab):
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)

# ...
# Définissez d'autres fonctions de coût Lab* si nécessaire

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

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-connexité
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

def find_shortest_path():
    global image, points, canvas, imgtk, graph
    if len(points) == 2:
        start, end = points
        path = dijkstra(image, start[::-1], end[::-1], cost_function_lab)

        # Dessiner le chemin sur l'image
        for i in range(len(path) - 1):
            cv2.line(image, path[i][::-1], path[i+1][::-1], (0, 255, 0), 2)

        # Mettre à jour l'image affichée dans Tkinter
        refresh_image()

        # Trouver le chemin dans le graphe
        shortest_path = []
        for point in path:
            shortest_path.append((point[1], point[0]))
        if has_path(graph, shortest_path[0], shortest_path[-1]):
            print("Chemin le plus court (sommets du graphe) :", shortest_path)
        else:
            print("Aucun chemin trouvé dans le graphe.")

def show_button():
    btn = Button(window, text="Trouver le chemin le plus court", command=find_shortest_path)
    btn.pack(side="bottom")

def on_canvas_click(event):
    global points, image, canvas, imgtk
    if len(points) < 2:
        x, y = event.x, event.y
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        refresh_image()
        if len(points) == 1:
            update_instructions(f"Point 1 sélectionné : ({x}, {y}). Veuillez sélectionner le point 2.")
        elif len(points) == 2:
            update_instructions(f"Point 2 sélectionné : ({x}, {y}). Cliquez sur le bouton pour calculer le chemin.")
            show_button()

def refresh_image():
    global canvas, image, window, imgtk
    image_for_tk = lab_to_rgb(image)
    im = Image.fromarray(image_for_tk)
    imgtk = ImageTk.PhotoImage(image=im)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

def has_path(graph, start, end):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            return True
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, []))

    return False

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

    # Charger l'image et vérifier si elle est chargée correctement
    image = cv2.imread('Mona_LisaColor.png')
    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return

    # Convertir l'image en Lab*
    image = rgb_to_lab(image)

    # Conversion pour affichage dans Tkinter
    image_for_tk = lab_to_rgb(image)
    im = Image.fromarray(image_for_tk)
    imgtk = ImageTk.PhotoImage(image=im)

    canvas = tk.Canvas(window, width=imgtk.width(), height=imgtk.height())
    canvas.pack(side="top", fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)
    canvas.bind("<Button-1>", on_canvas_click)

    # Créer le graphe représentant les connexions entre les pixels voisins
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            neighbors = []
            if x > 0:
                neighbors.append((y, x - 1))  # Connexions horizontales
            if y > 0:
                neighbors.append((y - 1, x))  # Connexions verticales
            graph[(y, x)] = neighbors

    print_graph(graph)  # Imprimez le graphe
    window.mainloop()

if __name__ == "__main__":
    main()