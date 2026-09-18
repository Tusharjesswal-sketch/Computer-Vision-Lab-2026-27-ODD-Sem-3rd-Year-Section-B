import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image = cv2.imread("image.jpg")

if image is None:
    print("ERROR: image.jpg not found!")
    print("Make sure image.jpg is in the same folder as main.py")
    exit()

print("Image loaded successfully!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

print("Grayscale conversion completed.")
print("Gaussian blur completed.")

_, global_threshold = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

print("Global Thresholding completed.")
otsu_value, otsu_threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu Thresholding completed.")
print("Optimal Otsu threshold value:", otsu_value)
adaptive_threshold = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

print("Adaptive Thresholding completed.")
_, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)
sure_bg = cv2.dilate(
    opening,
    kernel,
    iterations=3
)
dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)
_, sure_fg = cv2.threshold(
    dist_transform,
    0.7 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(
    sure_bg,
    sure_fg
)
_, markers = cv2.connectedComponents(
    sure_fg
)

markers = markers + 1

markers[unknown == 255] = 0
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)
watershed_image[markers == -1] = [255, 0, 0]

print("Watershed Segmentation completed.")
small_image = cv2.resize(
    image,
    (256, 256)
)
rgb_image = cv2.cvtColor(
    small_image,
    cv2.COLOR_BGR2RGB
)

pixels = rgb_image.reshape(
    (-1, 3)
)

pixels = np.float32(pixels)
k = 4

print("Applying K-Means with", k, "clusters...")

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(pixels)

centers = np.uint8(
    kmeans.cluster_centers_
)

segmented_pixels = centers[labels]

segmented_image = segmented_pixels.reshape(
    rgb_image.shape
)

print("K-Means Clustering completed.")

plt.figure(figsize=(14, 10))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Thresholding")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(watershed_image, cv2.COLOR_BGR2RGB))
plt.title("Watershed")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(segmented_image)
plt.title("K-Means Segmentation")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(rgb_image)
plt.title("Resized Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n======================================")
print("SEGMENTATION TECHNIQUES COMPARISON")
print("======================================")

print("""
1. Global Thresholding:
   - Uses one fixed threshold.
   - Simple and fast.
   - Works well when illumination is uniform.

2. Otsu's Thresholding:
   - Automatically selects an optimal threshold.
   - Useful when the image has a clear foreground/background separation.

3. Adaptive Thresholding:
   - Uses different threshold values for different regions.
   - Useful for images with uneven illumination.

4. Watershed:
   - Treats the image as a topographic surface.
   - Useful for separating touching or overlapping objects.

5. K-Means:
   - Groups pixels into clusters based on color.
   - Useful for color-based image segmentation.
""")

print("Experiment 6 completed successfully!")