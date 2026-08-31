import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
 

image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS9zfzfKGaBIjTUd2PVwAw0IeKmCyj9g4OrOmx5FMpHkw&s=10"
 
image = None
 
if image_path.startswith("http://") or image_path.startswith("https://"):
    # cv2.imread() cannot read directly from a web link, so we download
    # the image bytes first, then decode them into an image.
    try:
        print("Downloading image from URL...")
        req = urllib.request.Request(image_path, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=10)
        image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Could not decode the downloaded file as an image.")
        else:
            print("Loaded image from URL.")
    except Exception as e:
        print("Could not download image from URL:", e)
 
elif os.path.exists(image_path):
    # Read the image in grayscale mode (single channel, easier to filter)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print("Loaded image from:", image_path)
 
if image is None:
    # If no image is found, we create a simple test image ourselves.
    # This has shapes (so edges exist) and random noise (so filters have
    # something to clean up). This makes sure the code always runs.
    print("No input image found. Creating a sample test image instead.")
    image = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 200, -1)     # a filled square
    cv2.circle(image, (220, 220), 60, 255, -1)               # a filled circle
    cv2.line(image, (0, 250), (300, 250), 180, 4)            # a line
 
    # Add "salt-and-pepper" noise (random black and white dots)
    noise = np.random.randint(0, 100, image.shape)
    image[noise < 3] = 0      # pepper (black dots)
    image[noise > 97] = 255   # salt (white dots)
 
# ----------------------------------------------------------------------
# STEP 2: Display the original image (we will show it later in the grid)
# ----------------------------------------------------------------------
# (Handled together with all other results in Step 9 for easy comparison)
 
# ----------------------------------------------------------------------
# STEP 3: Gaussian Blur (Low-Pass Filter)
# ----------------------------------------------------------------------
# Smooths the image using a weighted average -> pixels closer to the
# center of the kernel matter more. Good general-purpose noise reducer.
gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0)
 
# ----------------------------------------------------------------------
# STEP 4: Median Filtering (Low-Pass Filter)
# ----------------------------------------------------------------------
# Replaces each pixel with the MEDIAN value of its neighborhood.
# Very effective against salt-and-pepper noise because extreme values
# (very black or very white dots) get ignored.
median_filtered = cv2.medianBlur(image, 5)
 
# ----------------------------------------------------------------------
# STEP 5: Average (Mean) Filtering (Low-Pass Filter)
# ----------------------------------------------------------------------
# Replaces each pixel with the plain AVERAGE of its neighborhood.
# Simple and fast, but blurs edges more than Gaussian blur.
average_filtered = cv2.blur(image, (5, 5))
 
# ----------------------------------------------------------------------
# STEP 6: Laplacian Filtering (High-Pass Filter)
# ----------------------------------------------------------------------
# Highlights regions of rapid intensity change (edges) in all directions
# at once, using the second derivative of the image.
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian_display = cv2.convertScaleAbs(laplacian)  # convert back to 0-255 for display
 
# ----------------------------------------------------------------------
# STEP 7: Sobel Edge Detection (High-Pass Filter)
# ----------------------------------------------------------------------
# Detects edges by estimating the gradient (rate of change) of intensity.
# Sobel X finds vertical edges; Sobel Y finds horizontal edges.
sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel_x_display = cv2.convertScaleAbs(sobel_x)
sobel_y_display = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
 
# ----------------------------------------------------------------------
# STEP 8 & 9: Compare and display all results together
# ----------------------------------------------------------------------
titles = [
    "Original Image",
    "Gaussian Blur (Low-Pass)",
    "Median Filter (Low-Pass)",
    "Average Filter (Low-Pass)",
    "Laplacian (High-Pass)",
    "Sobel X (High-Pass)",
    "Sobel Y (High-Pass)",
    "Sobel Combined (High-Pass)",
]
 
images_to_show = [
    image,
    gaussian_blur,
    median_filtered,
    average_filtered,
    laplacian_display,
    sobel_x_display,
    sobel_y_display,
    sobel_combined,
]
 
plt.figure(figsize=(14, 7))
for i in range(len(images_to_show)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images_to_show[i], cmap="gray")
    plt.title(titles[i], fontsize=10)
    plt.axis("off")
 
plt.tight_layout()
plt.savefig("filtering_results.png", dpi=150)
print("Saved comparison figure as filtering_results.png")
plt.show()
 
# ----------------------------------------------------------------------
# STEP 10: Simple observations (printed to console)
# ----------------------------------------------------------------------
print("\n--- Observations ---")
print("Low-Pass Filters (Gaussian, Median, Average): reduce noise but blur edges.")
print("Median filter works best on salt-and-pepper noise (dots), keeps edges sharper than Average filter.")
print("High-Pass Filters (Laplacian, Sobel): highlight edges but amplify noise.")
print("Sobel X detects vertical edges; Sobel Y detects horizontal edges.")