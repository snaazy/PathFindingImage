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

# Exemple de fonctions de calcul de coût 


def cost_function_lab(point1, point2, image_lab):
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def cost_function_labDif(point1, point2, image_lab):
    L1, a1, b1 = image_lab[point1[0], point1[1]].astype(int)
    L2, a2, b2 = image_lab[point2[0], point2[1]].astype(int)
    return abs(L1 - L2)

def cost_function_rgb(point1, point2, image):
    pixel1 = image[point1[0], point1[1]].astype(int)
    pixel2 = image[point2[0], point2[1]].astype(int)
    return np.linalg.norm(pixel1 - pixel2)

def cost_function_manhattan(point1, point2, image):
    pixel1 = image[point1[0], point1[1]].astype(int)
    pixel2 = image[point2[0], point2[1]].astype(int)
    return np.sum(np.abs(pixel1 - pixel2))

def cost_function_combined(point1, point2, image, alpha=1.0, beta=1.0):
    pixel1 = image[point1[0], point1[1]].astype(int)
    pixel2 = image[point2[0], point2[1]].astype(int)
    color_cost = np.linalg.norm(pixel1 - pixel2)
    spatial_cost = np.linalg.norm(np.array(point1) - np.array(point2))
    total_cost = alpha * color_cost + beta * spatial_cost
    return total_cost








# ----------------------------------------------------------------------
""" def dijkstra(image_lab, start, end):
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

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x = current_node[1] + dx
                new_y = current_node[0] + dy
                neighbor = (new_y, new_x)

                if 0 <= new_x < width and 0 <= new_y < height and not visited[neighbor]:
                    cost = cost_function_lab(current_node, neighbor, image_lab)
                    new_dist = dist + cost
                    if new_dist < distance_map[neighbor]:
                        distance_map[neighbor] = new_dist
                        parent_map[neighbor] = current_node
                        heapq.heappush(priority_queue, (new_dist, neighbor))

    path = [end]
    while path[-1] != start:
        parent = tuple(parent_map[path[-1]])
        path.append(parent)

    return path[::-1]  """

 
def dijkstra(image_lab, start, end):
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
                cost = cost_function_lab(current_node, neighbor, image_lab)
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
    global image, points, canvas, imgtk
    if len(points) == 2:
        start, end = points
        path = dijkstra(image, start[::-1], end[::-1])

        # Dessiner le chemin sur l'image
        for i in range(len(path) - 1):
            cv2.line(image, path[i][::-1], path[i+1][::-1], (0, 255, 0), 2)

        # Mettre à jour l'image affichée dans Tkinter
        refresh_image()

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
    image_for_tk = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    im = Image.fromarray(image_for_tk)
    imgtk = ImageTk.PhotoImage(image=im)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

def main():
    global image, window, canvas, instruction_label, imgtk
    window = tk.Tk()
    window.title("Trouver le chemin le plus court")

    instruction_label = Label(window, text="Veuillez sélectionner le point 1.")
    instruction_label.pack(side="top")

    # Charger l'image et vérifier si elle est chargée correctement
    image = cv2.imread('road.png')
    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return

    # Conversion pour affichage dans Tkinter
    image_for_tk = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image_for_tk)
    imgtk = ImageTk.PhotoImage(image=im)

    canvas = tk.Canvas(window, width=imgtk.width(), height=imgtk.height())
    canvas.pack(side="top", fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)
    canvas.bind("<Button-1>", on_canvas_click)

    window.mainloop()

if __name__ == "__main__":
    main()
