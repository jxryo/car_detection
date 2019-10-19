class_names = {
    1: 'car',
    0: 'no car'
}
import matplotlib.pyplot  as plt


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # print(img.shape)
    plt.imshow(img.reshape(11, 12))
    # predicted_label = np.argmax(predictions_array)
    predicted_label = predictions_array
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
    plt.show()


import pickle
import time
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

# get data
dist_pickle = pickle.load(open("train_data.p", "rb"))

# split train test
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = tts(dist_pickle['X'], dist_pickle['y'], test_size=0.2, random_state=rand_state)
# clf
# #LinearSVC 0.98...
# clf = svm.LinearSVC()
# t1 = time.time()
# clf.fit(X_train,y_train)
# t2 = time.time()
# print(round(t2-t1,2),'Seconds to train classfier...')
# # Check the score of the SVC
# print('Test Accuracy of classfier = ', round(clf.score(X_test, y_test), 4))
# #Check the prediction time for a single sample
# t1 = time.time()
# #100 number samples
# n_predict = 1000
# print('My classifier predicts:',clf.predict(X_test[0:n_predict]))
# print('For these',n_predict, 'labels: ', y_test[0:n_predict])
# t2 = time.time()
# print(round(t2-t1, 5), 'Seconds to predict', n_predict,'labels with classfier')

# #DecisionTreeClassifier 0.91...
# clf = tree.DecisionTreeClassifier()
# t1 = time.time()
# clf.fit(X_train,y_train)
# t2 = time.time()
# print(round(t2-t1,2),'Seconds to train classfier...')
# # Check the score of the SVC
# print('Test Accuracy of classfier = ', round(clf.score(X_test, y_test), 4))
# #Check the prediction time for a single sample
# t1 = time.time()
# #100 number samples
# n_predict = 1000
# print('My classifier predicts:',clf.predict(X_test[0:n_predict]))
# print('For these',n_predict, 'labels: ', y_test[0:n_predict])
# t2 = time.time()
# print(round(t2-t1, 5), 'Seconds to predict', n_predict,'labels with classfier')


# RandomForestClassifier 0.95...
clf = RandomForestClassifier()
t1 = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t1, 2), 'Seconds to train classfier...')
# Check the score of the SVC
print('Test Accuracy of classfier = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t1 = time.time()
# 100 number samples
n_predict = 1000

# for ix in range(n_predict):
#     plot_image(ix, clf.predict(X_test[0:n_predict]), y_test[0:n_predict], X_test[0:n_predict])

print('My classifier predicts:', clf.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t1, 5), 'Seconds to predict', n_predict, 'labels with classfier')
