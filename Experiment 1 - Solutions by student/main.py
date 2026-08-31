import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import os
import urllib.request
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTkx_E6utONwNqlCR1THDHd69fvefzJjAbIShwMvMlIaPRGg6IdMbIYQAE&s=10"
image_path = "sample.jpg"  # local filename it will be saved as
if not os.path.exists(image_path):
    urllib.request.urlretrieve(image_url, image_path)
    print(f"Downloaded image to {image_path}")
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(
        f"Could not load image at '{image_path}'. "
        "Place an image in the working directory or update the path."
    )
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis("off")
plt.savefig("01_original.png", bbox_inches="tight")
plt.close()
height, width, channels = img_bgr.shape
print("----- Image Properties -----")
print(f"Dimensions (H x W): {height} x {width}")
print(f"Number of Channels: {channels}")
print(f"Data Type: {img_bgr.dtype}")
print(f"Total Pixels: {img_bgr.size}")
print(f"Shape: {img_bgr.shape}")
cv2.imwrite("output_image.jpg", img_bgr)   # JPEG (lossy)
cv2.imwrite("output_image.png", img_bgr)   # PNG (lossless)
jpg_size = os.path.getsize("output_image.jpg")
png_size = os.path.getsize("output_image.png")
print("\n----- File Size Comparison -----")
print(f"JPEG size: {jpg_size} bytes")
print(f"PNG size: {png_size} bytes")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original (RGB)")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(img_hsv)
plt.title("HSV")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(img_lab)
plt.title("LAB")
plt.axis("off")
plt.tight_layout()
plt.savefig("02_color_spaces.png", bbox_inches="tight")
plt.close()
img_resized = cv2.resize(img_bgr, (300, 300))
img_rotated_90 = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
(h, w) = img_bgr.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
img_rotated_45 = cv2.warpAffine(img_bgr, rotation_matrix, (w, h))
img_flipped_h = cv2.flip(img_bgr, 1)   # Horizontal flip
img_flipped_v = cv2.flip(img_bgr, 0)   # Vertical flip
plt.figure(figsize=(14, 8))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title("Resized (300x300)")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img_rotated_90, cv2.COLOR_BGR2RGB))
plt.title("Rotated 90°")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(img_rotated_45, cv2.COLOR_BGR2RGB))
plt.title("Rotated 45°")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(img_flipped_h, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(img_flipped_v, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")
plt.tight_layout()
plt.savefig("03_geometric_transforms.png", bbox_inches="tight")
plt.close()
img_negative = 255 - img_bgr
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_negative, cv2.COLOR_BGR2RGB))
plt.title("Negative (Complement)")
plt.axis("off")
plt.tight_layout()
plt.savefig("04_negative.png", bbox_inches="tight")
plt.close()
y1, y2, x1, x2 = 50, 200, 50, 200
img_roi = img_bgr[y1:y2, x1:x2]
print("\n----- ROI Properties -----")
print(f"ROI Shape: {img_roi.shape}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))
plt.title("Cropped ROI")
plt.axis("off")
plt.tight_layout()
plt.savefig("05_roi.png", bbox_inches="tight")
plt.close()
titles = ["Original", "Grayscale", "HSV", "LAB",
          "Resized", "Rotated 45°", "H-Flip", "Negative", "ROI"]
images = [img_rgb, img_gray, img_hsv, img_lab,
          cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(img_rotated_45, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(img_flipped_h, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(img_negative, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)]
plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    cmap = "gray" if titles[i] == "Grayscale" else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.savefig("06_final_comparison.png", bbox_inches="tight")
plt.close()
print("\nAll operations completed successfully.")
print("Check the file explorer for: 01_original.png, 02_color_spaces.png,")
print("03_geometric_transforms.png, 04_negative.png, 05_roi.png, 06_final_comparison.png")