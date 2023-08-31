import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

def diagonal_translation(picture):
    img = Image.fromarray(picture.astype('uint8'), 'RGB')
    w = int(img.size[0] * 0.75)
    h = int(img.size[1] * 0.75)
    border = (15, 15, img.size[0] - w - 15, img.size[1] - h - 15)
    img = img.resize((w, h), Image.LANCZOS)
    translated = ImageOps.expand(img, border=border, fill='black')
    return np.array(translated)

# Load an example image
img_path = 'dataset/FERG_DB_256/aia/aia_anger/aia_anger_1.png'
img = Image.open(img_path)

# Convert the image to a numpy array
img_arr = np.array(img)

# Apply the diagonal translation
translated_img_arr = diagonal_translation(img_arr)

# Convert the translated image array back to a PIL image
translated_img = Image.fromarray(translated_img_arr)

def undo_diagonal_translation(picture, original_size):
    img = Image.fromarray(picture.astype('uint8'), 'RGB')
    w, h = original_size
    x1, y1, x2, y2 = 15, 15, 15 + w, 15 + h  # compute original image coordinates
    cropped = img.crop((x1, y1, x2, y2))  # crop image
    resized = cropped.resize((img.size[0], img.size[1]), Image.LANCZOS) #resize the cropped image to original dimensions
    return np.array(resized)

original_size = (100, 100)  # original image size
picture = np.array(translated_img_arr)  # transformed image
picture = undo_diagonal_translation(picture, original_size)  # undo diagonal translation

# Show the original and translated images side-by-side
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original')
ax[1].imshow(translated_img)
ax[1].set_title('Translated')
ax[2].imshow(picture)
ax[2].set_title('Undo Translated')

plt.show()