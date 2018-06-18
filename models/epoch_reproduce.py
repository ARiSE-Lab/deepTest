"""
This is an example script for reproducing epoch model in predicting hmb3 dataset
and udacity autonomous car challenge2 test dataset.
"""
from __future__ import print_function
import os
import argparse
import csv
import cv2
import numpy as np
from epoch_model import build_cnn, build_InceptionV3
from keras import backend as K
from scipy.misc import imresize


def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def exact_output(y):
    return y

def normalize_input(x):
    return x / 255.


def preprocess_image(image, image_size):

    img = image
        # Cropping
    crop_img = img[200:, :]
        # Resizing
    img = imresize(crop_img, size=image_size)
    imgs = []
    imgs.append(img)
    if len(imgs) < 1:
        print('Error no image at timestamp')

    img_block = np.stack(imgs, axis=0)
    if K.image_dim_ordering() == 'th':
        img_block = np.transpose(img_block, axes=(0, 3, 1, 2))
    return img_block

def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print("yhat and label have different lengths")
        return -1
    for i in xrange(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        #print(predicted_steering)
        #print(steering)
        mse += (float(steering) - float(predicted_steering))**2.
    return (mse/count) ** 0.5

def epoch_reproduce(dataset_path):
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    model_name = "cnn"
    image_size = (128, 128)
    weights_path = 'weights_HMB_2.hdf5' # Change to your model weights

    # Model build
    # ---------------------------------------------------------------------------------
    model_builders = {
        'V3': (build_InceptionV3, preprocess_input_InceptionV3, exact_output)
              , 'cnn': (build_cnn, normalize_input, exact_output)}

    if model_name not in model_builders:
        raise ValueError("unsupported model %s" % model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    model = model_builder(image_size, weights_path)
    print('model %s built...' % model_name)

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
        seed_image = cv2.imread(os.path.join(seed_inputs1, f))
        seed_image = preprocess_image(seed_image, image_size)
        test_x = input_processor(seed_image.astype(np.float32))
        yhat = model.predict(test_x)
        yhat = yhat[0][0]
        yhats.append(yhat)
        labels.append(truth[f])
        if count % 500 == 0:
            print("processed images: " + str(count) + " total: " + str(total))
        count = count + 1

    for f in filelist2:
        seed_image = cv2.imread(os.path.join(seed_inputs2, f))
        seed_image = preprocess_image(seed_image, image_size)
        test_x = input_processor(seed_image.astype(np.float32))
        yhat = model.predict(test_x)
        yhat = yhat[0][0]
        yhats.append(yhat)
        labels.append(truth[f])
        if count % 500 == 0:
            print("processed images: " + str(count) + " total: " + str(total))
        count = count + 1
    mse = calc_rmse(yhats, labels)
    print("mse: " + str(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    epoch_reproduce(args.dataset)
