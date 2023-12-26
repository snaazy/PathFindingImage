## Université Paris-Cité

Auteurs du sujet : LOMÉNIE Nicolas, MAHÉ Gael, élaboré avec LOBRY Sylvain.

# Trouver le chemin le plus court dans une image

Ce projet consiste en la création d'une interface graphique permettant de sélectionner deux pixels dans une image et de trouver le chemin le plus court entre ces deux pixels dans le contexte d'un graphe non orienté valué, en utilisant l'algorithme de Dijkstra.

## Prérequis

Assurez-vous d'avoir installé les bibliothèques Python suivantes :
- `cv2` (OpenCV)
- `numpy`
- `tkinter` (pour l'interface graphique)
- `PIL` (Pillow)

Vous pouvez les installer en utilisant `pip` :
```bash
pip install opencv-python numpy pillow
```


## Utilisation

1. Exécutez le script Python `main.py` (une fois dans le dossier /src).
2. Une fenêtre s'ouvrira avec une interface graphique.
3. Cliquez sur le bouton "Choisir une image" pour sélectionner une image depuis votre ordinateur.
4. Sélectionnez deux points en cliquant sur l'image. Le premier point sera le point de départ, et le deuxième sera le point d'arrivée.
5. Cliquez sur le bouton "Trouver le chemin le plus court" pour calculer le chemin le plus court entre les deux points en utilisant l'algorithme de Dijkstra.

## Visualisation du chemin

Lorsque le plus court chemin est calculé, il est affiché de manière progressive sur l'image selectionnée. Chaque segment du chemin est coloré en fonction de l'intensité moyenne des pixels le long du segment, ce qui permet une visualisation détaillée des variations d'intensité sur le chemin.

## Fonctions de coût

Le projet propose différentes fonctions de coût pour évaluer la différence entre les pixels voisins lors du calcul du chemin le plus court. Vous pouvez changer la fonction de coût utilisée en modifiant l'appel à la fonction `dijkstra` dans la fonction `find_shortest_path` du script `main.py`. Les fonctions de coût disponibles sont :

- `cost_function_lab`: Utilise la différence de couleur et de luminosité entre les pixels en mode Lab*.
- `cost_function_labDif`: Utilise uniquement la différence de luminosité (L) en mode Lab*.
- `cost_function_intensity`: Utilise la différence d'intensité lumineuse normalisée (en niveau de gris).
- `cost_function_local_contrast`: Utilise le contraste local autour des pixels.

## Réinitialisation

- Vous pouvez réinitialiser la sélection des points en cliquant sur le bouton "Réinitialiser".
- Vous pouvez également recharger l'image en utilisant le bouton "Choisir une autre image".

## Auteur

FEKIH HASSEN Yassine


