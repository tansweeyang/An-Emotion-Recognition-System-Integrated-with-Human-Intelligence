import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'dataset/FERG_DB_256/bonnie/bonnie_joy/bonnie_joy_1.png'


# Function to generate a feature map with a given standard deviation
def generate_feature_map(image_path, std_dev):
    # Load an image from file
    image = cv2.imread(image_path)

    # Resize the image to the desired size
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)

    # Determine the number of color channels (3 for RGB, 1 for grayscale)
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Generate noise with the same shape as the image
    noise = np.random.normal(0, std_dev, image_size + (num_channels,))

    # Add the noise to the image
    feature_map = image + noise

    return feature_map


# Parameters
image_size = (128, 128)  # Size of the image
std_dev1 = 20.0  # Standard deviation for the first subplot
std_dev2 = 70.0  # Standard deviation for the second subplot

# Generate the feature maps
feature_map1 = generate_feature_map(image_path, std_dev1)
feature_map2 = generate_feature_map(image_path, std_dev2)

# Create subplots
plt.figure(figsize=(12, 5))

# First subplot
plt.subplot(1, 2, 1)
plt.imshow(feature_map1, cmap='viridis')
plt.title(f'Std Dev = {std_dev1}')
plt.colorbar()

# Second subplot
plt.subplot(1, 2, 2)
plt.imshow(feature_map2, cmap='viridis')
plt.title(f'Std Dev = {std_dev2}')
plt.colorbar()

plt.suptitle('Effect of Increasing Standard Deviation on Feature Maps')
plt.tight_layout()
plt.show()
