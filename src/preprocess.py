from PIL import Image
import numpy as np

img = Image.open("src/image.png").convert("L")  # grayscale
img = img.resize((28, 28))
pixels = np.array(img).flatten() / 255.0

np.savetxt("data/image.txt", pixels)
