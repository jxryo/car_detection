from sklearn.model_selection import train_test_split as tts
import glob
import time
import utils
import numpy as np
import pickle

# divide car no car
noncars = glob.glob('food/non-vehicles/*/*.png')
cars = glob.glob('food/vehicles/*/*.png')

# set values
colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 32
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
# set feature
t1 = time.time()

car_features = utils.extract_features(cars, cspace=colorspace, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                      hog_channel=hog_channel)
notcar_features = utils.extract_features(noncars, cspace=colorspace, orient=orient,
                                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                         hog_channel=hog_channel)

t2 = time.time()
print(round(t2 - t1, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features))
X = X.astype(np.float64)

# A = np.vstack((cars,noncars))
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

data = {
    'X': X,
    'y': y
}
output = open('train_data.p', 'wb')
pickle.dump(data, output)
