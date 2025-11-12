import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Obtener ruta de la carpeta de imágenes
images_path = Path(__file__).parent.parent / "images"

# Buscar todas las imágenes de nemo
nemo_files = sorted(images_path.glob("nemo*.jpg"))

print(f"Se encontraron {len(nemo_files)} imágenes de Nemo")

# Número de clusters (colores)
k = 3

def distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt(np.sum((p1 - p2)**2))

def assign_clusters(X, clusters, k):
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

def update_clusters(X, clusters, k):
    """Actualiza los centros de los clusters"""
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
        
        clusters[i]['points'] = []
    
    return clusters

def pred_cluster(X, clusters, k):
    """Predice el cluster para cada pixel"""
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    
    return np.array(pred)

def kmeans_image(image_path, k, max_iterations=10):
    """Aplica k-means a una imagen"""
    # Leer imagen
    nemo = cv2.imread(str(image_path))
    nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
    
    # Convertir imagen a array de pixels
    h, w, c = nemo_rgb.shape
    X = nemo_rgb.reshape((h * w, 3)).astype(np.float32)
    
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
    
    # Ejecutar k-means iterativamente
    for iteration in range(max_iterations):
        clusters = assign_clusters(X, clusters, k)
        clusters = update_clusters(X, clusters, k)
    
    # Predecir clusters finales
    pred = pred_cluster(X, clusters, k)
    
    # Reconstruir imagen segmentada
    segmented = np.zeros_like(X)
    for i in range(k):
        segmented[pred == i] = clusters[i]['center']
    
    # Reshape de vuelta a imagen
    segmented_img = segmented.reshape((h, w, 3)).astype(np.uint8)
    pred_img = pred.reshape((h, w))
    
    return nemo_rgb, segmented_img, pred_img, clusters

# Procesar todas las imágenes
original_images = []
segmented_images = []
pred_images = []

for nemo_file in nemo_files:
    print(f"Procesando {nemo_file.name}...")
    nemo_rgb, segmented_img, pred_img, clusters = kmeans_image(nemo_file, k)
    original_images.append(nemo_rgb)
    segmented_images.append(segmented_img)
    pred_images.append(pred_img)

# Visualizar resultados en dos filas
fig, axs = plt.subplots(2, len(nemo_files), figsize=(18, 8))

# Si solo hay una imagen, axs no es una matriz bidimensional
if len(nemo_files) == 1:
    axs = axs.reshape(2, 1)

# Primera fila: Imágenes originales
for idx, original_img in enumerate(original_images):
    axs[0, idx].imshow(original_img)
    axs[0, idx].set_title(f"Original - {nemo_files[idx].name}")
    axs[0, idx].axis('off')

# Segunda fila: Imágenes segmentadas por k-means
for idx, segmented_img in enumerate(segmented_images):
    axs[1, idx].imshow(segmented_img)
    axs[1, idx].set_title(f"K-Means (k={k}) - {nemo_files[idx].name}")
    axs[1, idx].axis('off')

plt.suptitle(f"K-Means Clustering en 6 Imágenes de Nemo (k={k})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nProceso completado.")
