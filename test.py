from sklearn.model_selection import train_test_split as tts
import glob
from skimage import io
import time
import utils
import numpy as np
import pickle
import matplotlib.pyplot as plt
# divide car no car
#cars = glob.glob('food/vehicles/GTI_Far/*.png')


cars = glob.glob('food/non-vehicles/GTI/*.png')

# print(cars[0])
# set values
colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 32
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
# set feature
car_features = utils.extract_features(cars, cspace=colorspace, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                      hog_channel=hog_channel)
res_image = np.vstack((car_features))
res_image = res_image.astype(np.float64)

# plt.imshow(res_image[0])
print(res_image[0])
for i in range(5):
    plt.hist(res_image[i],bins=15)
# plt.xlabel('vehicle')
# plt.savefig('0000_c1.png')
plt.xlabel('non-vehicle')
plt.savefig('0000_c2.png')
# plt.show()
# plt.imshow(io.imread(cars[0]))
# plt.savefig('0000_n.png')