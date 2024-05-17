import os
import numpy as np
from PIL import Image

# path with TIF files from timelapse
path = "/Users/jamesbarrett/Downloads/FRAP/"

for file in os.listdir(path):
    if file.endswith(".tif"):
        img_array = np.array(Image.open(path+file)) # convert the image to a numpy array based on intensity
        np.savetxt(path+file[:-4]+".txt", img_array, newline='\n', fmt='%s') # export to text file

exit()