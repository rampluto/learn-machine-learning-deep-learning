import torch
import torch.nn as nn
import random
import cv2
import numpy as np
from PIL import Image

img = torch.tensor([[ [float(random.randint(0,15)) for col in range(4)] for col in range(16)] for row in range(16)])

conv = nn.Conv2d(16, 16, 4)
output = conv(img)


print(img)
print(output)
print(output.shape)
print(img.shape)
