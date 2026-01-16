from PIL import Image
import numpy as np

img = Image.open("data/image.png").convert("L") 

# converts image to gray scale 28 * 28
img = img.resize((28, 28))
pixels = np.array(img).flatten()  # raw values 0-255

np.savetxt("data/image.txt", pixels, fmt="%d")
