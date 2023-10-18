import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import cv2 

directory_path = './input_data/HE176392'
images = os.listdir(directory_path)

def compute_eigenfaces(images):
    flattened_images = []
    for img_file in images:
        img_path = os.path.join(directory_path, img_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        flattened_images.append(np.array(img).flatten())

    pca = PCA(n_components=5)
    pca.fit(flattened_images)

    eigenfaces = pca.components_.T

    return eigenfaces

# print(compute_eigenfaces(images))
images = compute_eigenfaces(images)
print(images)
images = np.array(images)[:,:1]
images = np.reshape(images, (640, 480))
images = images.astype('uint8')
print(images)
print(images.shape)
cv2.imshow('', images)
cv2.waitKey(0)
cv2.destroyAllWindows()
