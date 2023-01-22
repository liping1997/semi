import cv2
import os
import numpy as np
for i in os.listdir('./data/fundus'):

    img=cv2.imread('./data/fundus/{}'.format(i))

    print(i,np.max(img[:,768:1536,:]))