import cv2
import sys

img = cv2.imread(sys.argv[1])

# Convert to grayscale Increase contrast using CLAHE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# Apply Gaussian blur and Sobel filter
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.addWeighted(sobel_x, 0.008, sobel_y, 0.008, 0)

# Apply thresholding
ret, img_thresh = cv2.threshold(gray_clahe, 182, 255, cv2.THRESH_BINARY)

cv2.imshow("Cool gray", img_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("/Users/monkey/Public/Python/Diploma_1/Main/filtered.jpg", img_sobel)


# Display the result
# img_sobel
# img_thresh
# gray_clahe
