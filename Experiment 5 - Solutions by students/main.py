import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

image_path = "image.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: image.jpg was not found!")
    print("Make sure image.jpg is inside the same folder as main.py")
    exit()

print("Image loaded successfully!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Image converted to grayscale.")

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("\n========== SIFT RESULTS ==========")
print("Number of keypoints:", len(keypoints))

if descriptors is not None:
    print("Descriptor shape:", descriptors.shape)
else:
    print("No descriptors detected.")
sift_image = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


gray_resized = cv2.resize(gray, (256, 256))

hog_features, hog_image = hog(
    gray_resized,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm="L2-Hys"
)

hog_image = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)

print("\n========== HOG RESULTS ==========")
print("Number of HOG features:", len(hog_features))
print("HOG feature vector shape:", hog_features.shape)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n========== SIFT vs HOG ==========")

print("SIFT:")
print("- Detects important keypoints.")
print("- Produces descriptors for image matching.")
print("- Designed to handle scale and rotation changes.")

print("\nHOG:")
print("- Represents image information using gradient orientations.")
print("- Useful for describing object shapes and edges.")
print("- Commonly useful for object detection.")

print("\nExperiment completed successfully!")
print("\n========== IMAGE MATCHING ==========")

image2 = cv2.imread("image2.jpg")

if image2 is None:
    print("ERROR: image2.jpg was not found!")
    exit()

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

if descriptors1 is None or descriptors2 is None:
    print("Not enough features for matching.")
    exit()
FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)
matches = flann.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)
good_matches = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))
match_image = cv2.drawMatches(
    image,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(15, 8))

plt.imshow(
    cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
)

plt.title("SIFT Image Matching")
plt.axis("off")

plt.show()

print("\nImage matching completed!")

print("\n========== STRENGTHS AND LIMITATIONS ==========")

print("""
SIFT:
Strengths:
- Scale invariant
- Rotation invariant
- Robust for image matching
- Useful for object recognition

Limitations:
- Computationally more expensive
- Can require more processing time
- Less suitable for very large-scale real-time applications

HOG:
Strengths:
- Good representation of object shape
- Useful for edge and gradient information
- Commonly used for object detection

Limitations:
- Not inherently scale invariant
- Sensitive to changes in viewpoint
- Less effective when object appearance changes significantly
""")