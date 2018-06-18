"""
This is an example script for reproducing rambo model in predicting hmb3 dataset
and udacity autonomous car challenge2 test dataset.
"""
from __future__ import print_function
import sys
import os
import argparse
import csv
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
import cv2

reload(sys)
sys.setdefaultencoding('ISO-8859-1')

class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        print (self.mean_angle)
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        #img_path = 'test.jpg'
        #misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:, :, ::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
            return self.model.predict(X)[0]

def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print ("yhat and label have different lengths")
        return -1
    for i in xrange(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        #print(predicted_steering)
        #print(steering)
        mse += (float(steering) - float(predicted_steering))**2.
    return (mse/count) ** 0.5

def rambo_reproduce(dataset_path):
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    model = Model("./final_model.hdf5", "./X_train_mean.npy")
    filelist1 = []
    for image_file in sorted(os.listdir(seed_inputs1)):
        if image_file.endswith(".jpg"):
            filelist1.append(image_file)
    truth = {}
    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]
    for i in label1:
        truth[i[0]+".jpg"] = i[1]

    filelist2 = []
    for image_file in sorted(os.listdir(seed_inputs2)):
        if image_file.endswith(".jpg"):
            filelist2.append(image_file)
    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]

    for i in label2:
        truth[i[0]+".jpg"] = i[1]

    yhats = []
    labels = []
    count = 0
    total = len(filelist1) + len(filelist2)
    for f in filelist1:
        yhat = model.predict(os.path.join(seed_inputs1, f))
        yhats.append(yhat)
        labels.append(truth[f])
        if count % 500 == 0:
            print ("processed images: " + str(count) + " total: " + str(total))
        count = count + 1

    for f in filelist2:
        yhat = model.predict(os.path.join(seed_inputs2, f))
        yhats.append(yhat)
        labels.append(truth[f])
        if count % 500 == 0:
            print ("processed images: " + str(count) + " total: " + str(total))
        count = count + 1
    mse = calc_rmse(yhats, labels)
    print("mse: " + str(mse))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    rambo_reproduce(args.dataset)
