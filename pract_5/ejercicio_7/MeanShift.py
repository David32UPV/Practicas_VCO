import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from pathlib import Path

# Obtener ruta relativa a la imagen
image_path = Path(__file__).parent.parent / "images" / "nemo0.jpg"

# filter to reduce noise
img = cv2.medianBlur(cv2.imread(str(image_path)), 3)
# flatten the image
flat_image = img.reshape((-1,3))
flat_image = np.float32(flat_image)
# meanshift
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled=ms.labels_
# get number of segments
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])
# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
 total[label] = total[label] + flat_image[i]
 count[label] += 1
avg = total/count
avg = np.uint8(avg)
# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((img.shape))
# show the result
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()