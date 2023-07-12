import numpy as np
import matplotlib.pyplot as plt


with open('output.bin', 'rb') as f:
    img = np.frombuffer(f.read(), dtype=np.uint8).reshape(512, 512, 3)

print(img)
plt.imshow(img)
plt.show()
