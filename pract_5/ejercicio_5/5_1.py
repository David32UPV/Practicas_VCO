import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "nemo0.jpg"

# Leer imagen
nemo = cv2.imread(str(image_path))
nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

# Convertir imagen a array de pixels (n_pixels, 3)
# Reshape la imagen de (altura, ancho, 3) a (n_pixels, 3)
h, w, c = nemo_rgb.shape
X = nemo_rgb.reshape((h * w, 3)).astype(np.float32)

print(f"Datos de imagen: {X.shape}")

# Número de clusters (colores)
k = 3

# Inicializar clusters
clusters = {}
np.random.seed(23)

for idx in range(k):
    center = np.random.randint(0, 256, size=3).astype(np.float32)
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster

print(f"Centros iniciales:")
for i in clusters:
    print(f"  Cluster {i}: {clusters[i]['center']}")

def distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt(np.sum((p1 - p2)**2))

def assign_clusters(X, clusters):
    """Asigna cada pixel al cluster más cercano"""
    for idx in range(X.shape[0]):
        curr_x = X[idx]
        dist = []
        
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    
    return clusters

def update_clusters(X, clusters):
    """Actualiza los centros de los clusters"""
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
        
        clusters[i]['points'] = []
    
    return clusters

def pred_cluster(X, clusters):
    """Predice el cluster para cada pixel"""
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    
    return np.array(pred)

# Ejecutar k-means iterativamente
print("\nEjecutando k-Means...")
max_iterations = 10

for iteration in range(max_iterations):
    print(f"Iteración {iteration + 1}")
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(X, clusters)
    
    print(f"  Centros actualizados:")
    for i in clusters:
        print(f"    Cluster {i}: {clusters[i]['center']}")

# Predecir clusters finales
pred = pred_cluster(X, clusters)

# Reconstruir imagen segmentada
# Convertir predicciones a colores de los centros
segmented = np.zeros_like(X)
for i in range(k):
    segmented[pred == i] = clusters[i]['center']

# Reshape de vuelta a imagen (altura, ancho, 3)
segmented_img = segmented.reshape((h, w, 3)).astype(np.uint8)
pred_img = pred.reshape((h, w))

# Visualizar resultados
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original
axs[0].imshow(nemo_rgb)
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Segmentada por k-means
axs[1].imshow(segmented_img)
axs[1].set_title(f'K-Means (k={k})')
axs[1].axis('off')

# Mapa de clusters (coloreado por cluster ID)
cluster_map = plt.cm.get_cmap('tab10')(pred_img / k)
axs[2].imshow(cluster_map)
axs[2].set_title('Mapa de Clusters')
axs[2].axis('off')

plt.tight_layout()
plt.show()

print(f"\nColores finales de los clusters:")
for i in clusters:
    print(f"  Cluster {i}: RGB{tuple(clusters[i]['center'].astype(int))}")
