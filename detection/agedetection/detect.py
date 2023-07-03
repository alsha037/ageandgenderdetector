
from PIL import Image
pillow_image = Image.open(r"./image.jpg")
# pillow_image.show()

import cv2
import numpy as np

opencv_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
bw_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
