import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import rotate

# Load the image
original_image = mpimg.imread('dataset/FERG_DB_256/bonnie/bonnie_joy/bonnie_joy_1.png')

# Define transformation parameters
rotation_angle_90 = 90
rotation_angle_180 = 180
shift_pixels = 15  # Shift 15 pixels to the right and down

# Apply transformations
transformed_image_90 = rotate(original_image, rotation_angle_90)
transformed_image_180 = rotate(original_image, rotation_angle_180)

# Create a blank canvas for the shifted image
shifted_image = np.zeros_like(original_image)

# Perform diagonal translation by copying pixels manually
shifted_image[shift_pixels:, shift_pixels:] = original_image[:-shift_pixels, :-shift_pixels]

# Create a figure with four subplots (1 row, 4 columns)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot the original image on the left
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot the transformed images
axes[1].imshow(transformed_image_90)
axes[1].set_title('Rotate +90 degrees')
axes[1].axis('off')

axes[2].imshow(transformed_image_180)
axes[2].set_title('Rotate +180 degrees')
axes[2].axis('off')

axes[3].imshow(shifted_image)
axes[3].set_title('Shift 15 pixels right and down')
axes[3].axis('off')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.2)

# Display the plot
plt.show()
