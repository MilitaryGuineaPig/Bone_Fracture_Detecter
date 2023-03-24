import os
import cv2
import numpy as np

# Define directory containing images
img_dir = "/Users/monkey/Public/Python/Science Research 2022- Bone Fracture Detection.v9i.voc/valid_n"

# Define filters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Loop through all files in directory
for filename in os.listdir(img_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        # Load image
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)

        # Apply filters
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_clahe = clahe.apply(gray)

        # Apply unsharp mask filter
        blurred = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
        sharp = cv2.addWeighted(gray_clahe, 0.2, blurred, 0.05, 0)

        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        histeq = clahe.apply(sharp)

        # Apply Gaussian blur and Sobel filter
        blurred = cv2.GaussianBlur(sharp, (3, 3), 0)
        sobel_x = cv2.Sobel(gray_clahe, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_clahe, cv2.CV_64F, 0, 1, ksize=3)
        img_sobel = cv2.addWeighted(sobel_x, 2, sobel_y, 2, 0)

        ret, img_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        # Save modified image
        new_filename = filename
        new_path = os.path.join(img_dir, new_filename)
        cv2.imwrite(new_path, img_sobel)
