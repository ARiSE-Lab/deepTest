from __future__ import print_function
import os
import argparse
import numpy as np
import time
import random
from collections import deque
from keras import backend as K
from epoch_model import build_cnn, build_InceptionV3
from scipy.misc import imread, imresize, imsave
from scipy.misc import imshow
from ncoverage import NCoverage
import csv
import cv2
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def preprocess_input_InceptionV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def exact_output(y):
    return y

def normalize_input(x):
    return x / 255.

def read_image(image_file, image_size):

    img = imread(image_file)
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

def read_images(seed_inputs, seed_labels, image_size):

    img_blocks = []
    for file in os.listdir(seed_inputs):
        if file.endswith(".jpg"):
            img_block = read_image(os.path.join(seed_inputs, file), image_size)
            img_blocks.append(img_block)
    return img_blocks

def read_transformed_image(image, image_size):

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

def image_translation(img, params):
    if not isinstance(params, list):
        params = [params, params]
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):
    if not isinstance(params, list):
        params = [params, params]
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)                    # new_img = img*alpha + beta

    return new_img

def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur

def rotation(img, params):

    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params[0], 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_brightness1(img, params):
    w = img.shape[1]
    h = img.shape[0]
    if params > 0:
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if 255-img[xj, xi, 0] < params:
                    img[xj, xi, 0] = 255
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] + params
                if 255-img[xj, xi, 1] < params:
                    img[xj, xi, 1] = 255
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] + params
                if 255-img[xj, xi, 2] < params:
                    img[xj, xi, 2] = 255
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] + params
    if params < 0:
        params = params*(-1)
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if img[xj, xi, 0] - 0 < params:
                    img[xj, xi, 0] = 0
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] - params
                if img[xj, xi, 1] - 0 < params:
                    img[xj, xi, 1] = 0
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] - params
                if img[xj, xi, 2] - 0 < params:
                    img[xj, xi, 2] = 0
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] - params

    return img

def image_brightness2(img, params):
    beta = params
    b, g, r = cv2.split(img)
    b = cv2.add(b, beta)
    g = cv2.add(g, beta)
    r = cv2.add(r, beta)
    new_img = cv2.merge((b, g, r))
    return new_img


def epoch_guided(dataset_path):
    model_name = "cnn"
    image_size = (128, 128)
    threshold = 0.2
    weights_path = './weights_HMB_2.hdf5' # Change to your model weights

    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    new_inputs = "./new/"
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
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    truth = {}
    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]
    for i in label1:
        truth[i[0]+".jpg"] = i[1]


    newlist = []
    for file in sorted(os.listdir(new_inputs)):
        if file.endswith(".jpg"):
            newlist.append(file)

    flag = 0
    #flag:0 start from beginning
    #flag:1 initialize from pickle files

    '''
    Pickle files are used for continuing the search after rerunning the script.
    Delete all pkl files and generated images for starting from the beginnning.
    '''
    if os.path.isfile("epoch_covdict2.pkl") and \
            os.path.isfile("epoch_stack.pkl") and \
            os.path.isfile("epoch_queue.pkl") and \
            os.path.isfile("generated.pkl"):
        with open('epoch_covdict2.pkl', 'rb') as input:
            covdict = pickle.load(input)
        with open('epoch_stack.pkl', 'rb') as input:
            epoch_stack = pickle.load(input)
        with open('epoch_queue.pkl', 'rb') as input:
            epoch_queue = pickle.load(input)
        with open('generated.pkl', 'rb') as input:
            generated = pickle.load(input)
        flag = 1

    nc = NCoverage(model, threshold)

    if flag == 0:
        filewrite = "wb"
        epoch_queue = deque()
        epoch_stack = []
        generated = 0
    else:
        nc.set_covdict(covdict)
        filewrite = "ab"
        print("initialize from files and continue from previous progress")

    C = 0 # covered neurons
    P = 0 # covered percentage
    T = 0 # total neurons
    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness2, image_blur]
    params = []
    params.append(list(xrange(-50, 50)))
    params.append(list(map(lambda x: x*0.1, list(xrange(5, 20)))))
    params.append(list(map(lambda x: x*0.1, list(xrange(-5, 5)))))
    params.append(list(xrange(-30, 30)))
    params.append(list(map(lambda x: x*0.1, list(xrange(1, 20)))))
    params.append(list(xrange(-21, 21)))
    params.append(list(xrange(1, 11)))

    maxtrynumber = 10
    maximages = 200
    cache = deque()
    image_count = 0
    #load nc, generation, population
    with open('result/epoch_rq3_100_2.csv', filewrite, 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if flag == 0:
            writer.writerow(['id', 'seed image(root)', 'parent image', 'new generated image',
                             'number of generated images', 'total_covered', 'total_neurons',
                             'coverage_percentage', 'transformations', 'yhat', 'baseline', 'label'])
            #initialize population and coverage
            print("compute coverage of original population")
            input_images = xrange(1, 101)
            for i in input_images:
                j = i * 50
                epoch_queue.append(os.path.join(seed_inputs1, filelist1[j]))

        while len(epoch_queue) > 0:
            current_seed_image = epoch_queue[0]
            print(str(len(epoch_queue)) + " images are left.")
            if len(epoch_stack) == 0:
                epoch_stack.append(current_seed_image)
            image = cv2.imread(current_seed_image)
            test_x = read_transformed_image(image, image_size)
            test_x = input_processor(test_x.astype(np.float32))
            nc.update_coverage(test_x)
            baseline_yhat = model.predict(test_x)
                #image_count = 0
            while len(epoch_stack) > 0:
                try:

                    image_file = epoch_stack[-1]
                    print("current image in stack " + image_file)
                    image = cv2.imread(image_file)
                    new_generated = False
                    for i in xrange(maxtrynumber):

                        tid = random.sample([0,1,2,3,4,5,6], 2)
                        if len(cache) > 0:
                            tid[0] = cache.popleft()
                        transinfo = ""
                        new_image = image
                        for j in xrange(2):
                            transformation = transformations[tid[j]]
                            #random choose parameter
                            param = random.sample(params[tid[j]], 1)
                            param = param[0]
                            transinfo = transinfo + transformation.__name__ + ':' + str(param) + ';'
                            print("transformation " + transformation.__name__ + "  parameter " + str(param))
                            new_image = transformation(new_image, param)

                        new_x = read_transformed_image(new_image, image_size)

                        test_x = input_processor(new_x.astype(np.float32))
                        if nc.is_testcase_increase_coverage(test_x):
                            print("Generated image increases coverage and will be added to population.")
                            cache.append(tid[0])
                            cache.append(tid[1])
                            generated = generated + 1
                            #image_count = image_count + 1
                            name = os.path.basename(current_seed_image)+'_' + str(generated)+'.jpg'
                            name = os.path.join(new_inputs, name)
                            cv2.imwrite(name, new_image)
                            epoch_stack.append(name)

                            nc.update_coverage(test_x)
                            yhat = model.predict(test_x)
                            covered, total, p = nc.curr_neuron_cov()
                            C = covered
                            T = total
                            P = p
                            csvrecord = []
                            csvrecord.append(100-len(epoch_queue))
                            csvrecord.append(os.path.basename(current_seed_image))
                            if len(epoch_stack) >= 2:
                                parent = os.path.basename(epoch_stack[-2])
                            else:
                                parent = os.path.basename(current_seed_image)
                            child = os.path.basename(current_seed_image)+'_' + str(generated)+'.jpg'
                            csvrecord.append(parent)
                            csvrecord.append(child)
                            csvrecord.append(generated)
                            csvrecord.append(C)
                            csvrecord.append(T)
                            csvrecord.append(P)
                            csvrecord.append(transinfo)
                            csvrecord.append(yhat[0][0])
                            csvrecord.append(baseline_yhat[0][0])
                            csvrecord.append(truth[os.path.basename(current_seed_image)])
                            print(csvrecord)
                            writer.writerow(csvrecord)
                            new_generated = True
                            break
                        else:
                            print("Generated image does not increase coverage.")
                    if not new_generated:
                        epoch_stack.pop()

                    save_object(epoch_stack, 'epoch_stack.pkl')
                    save_object(epoch_queue, 'epoch_queue.pkl')
                    save_object(nc.cov_dict, 'epoch_covdict2.pkl')
                    save_object(generated, 'generated.pkl')

                except ValueError:
                    print("value error")
                    epoch_stack.pop()
                    save_object(epoch_stack, 'epoch_stack.pkl')
                    save_object(epoch_queue, 'epoch_queue.pkl')

            epoch_queue.popleft()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    epoch_guided(args.dataset)
