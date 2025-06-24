# Shows an image

import cv2
import matplotlib.pyplot as plt

# Load your image
image_path = 'images/test.jpg'
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print("Failed to load image. Make sure 'test.jpg' exists.")
else:
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show image using matplotlib
    plt.imshow(image_rgb)
    plt.title("Loaded Image")
    plt.axis('off')  # Hide axes
    plt.show()
