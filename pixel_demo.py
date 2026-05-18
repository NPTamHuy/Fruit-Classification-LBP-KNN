import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "custom_images/pr.jpg"

image = cv2.imread(image_path)
if image is None:
    print("Khong doc duoc anh!")
    exit()

# CONVERT TO GRAYSCALE
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# RESIZE FOR VISUALIZATION
gray = cv2.resize(gray, (20, 20))

# PRINT PIXEL MATRIX
print("\nMa tran pixel grayscale:\n")
print(gray)


# VISUALIZE MATRIX
plt.figure(figsize=(8, 8))
plt.imshow(gray, cmap='gray')

# Show pixel values
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):

        plt.text(
            j,
            i,
            str(gray[i, j]),
            ha='center',
            va='center',
            color='red',
            fontsize=8
        )

plt.title("Grayscale Pixel Matrix")
plt.axis("off")
plt.show()