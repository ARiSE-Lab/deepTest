from __future__ import print_function
import argparse
import numpy as np
import os
from epoch_model import build_cnn, build_InceptionV3
from scipy.misc import imread, imresize
from keras import backend as K
from ncoverage import NCoverage
import csv
import cv2

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

    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):

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
    new_img = cv2.add(img, beta)                                  # new_img = img*alpha + beta

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

def main():
    pass



def epoch_testgen_coverage(index, dataset_path):
    model_name = "cnn"
    image_size = (128, 128)
    threshold = 0.2
    weights_path = './weights_HMB_2.hdf5' # Change to your model weights


    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")
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

    filelist2 = []
    for file in sorted(os.listdir(seed_inputs2)):
        if file.endswith(".jpg"):
            filelist2.append(file)

    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]

    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]


    nc = NCoverage(model, threshold)
    index = int(index)
    #seed inputs
    with open('result/epoch_coverage_70000_images.csv', 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if index == 0:
            writer.writerow(['index', 'image', 'tranformation', 'param_name', 'param_value',
                             'threshold', 'covered_neurons', 'total_neurons',
                             'covered_detail', 'y_hat', 'label'])
        if index/2 == 0:
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)
            for i in input_images:
                j = i * 5
                csvrecord = []


                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = read_transformed_image(seed_image, image_size)

                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)


                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)

            print("seed input done")

        #Translation
        if index/2 >= 1 and index/2 <= 10:
            #for p in xrange(1,11):
            p = index/2
            params = [p*10, p*10]
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)

            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_translation(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)
                #seed_image = read_image(os.path.join(seed_inputs1, filelist1[j]),image_size)
                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('translation')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)

                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)
            print("translation done")

        #Scale
        if index/2 >= 11 and index/2 <= 20:
        #for p in xrange(1,11):
            p = index/2-10
            params = [p*0.5+1, p*0.5+1]

            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)

            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_scale(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)

                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue
                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('scale')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)


                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)

        print("scale done")

        #Shear
        if index/2 >= 21 and index/2 <= 30:
        #for p in xrange(1,11):
            p = index/2-20
        #for p in xrange(1,11):
            params = 0.1*p
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)

            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_shear(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)

                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue
                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('shear')
                csvrecord.append('factor')
                csvrecord.append(params)
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)


                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)
        print("sheer done")

        #Rotation
        if index/2 >= 31 and index/2 <= 40:
            p = index/2-30
            params = p*3
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)
            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_rotation(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)
                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('rotation')
                csvrecord.append('angle')
                csvrecord.append(params)
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)

                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)

        print("rotation done")

        #Contrast
        if index/2 >= 41 and index/2 <= 50:
            p = index/2 - 40
            params = 1 + p*0.2
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)
            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_contrast(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)

                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('contrast')
                csvrecord.append('gain')
                csvrecord.append(params)
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)

                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)

        print("contrast done")

        #Brightness
        if index/2 >= 51 and index/2 <= 60:
            p = index/2 - 50
            params = p * 10
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)
            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_brightness2(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)

                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue
                if j < 2:
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('brightness')
                csvrecord.append('bias')
                csvrecord.append(params)
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)


                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)
        print("brightness done")

        #blur
        if index/2 >= 61 and index/2 <= 70:
            p = index/2 - 60
            params = p
            if index%2 == 0:
                input_images = xrange(1, 501)
            else:
                input_images = xrange(501, 1001)
            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_blur(seed_image, params)
                seed_image = read_transformed_image(seed_image, image_size)

                #new_covered1, new_total1, result = model.predict_fn(seed_image)
                test_x = input_processor(seed_image.astype(np.float32))
                #print('test data shape:', test_x.shape)
                yhat = model.predict(test_x)
                #print('steering angle: ', yhat)

                ndict = nc.update_coverage(test_x)
                new_covered1, new_total1, p = nc.curr_neuron_cov()

                tempk = []
                for k in ndict.keys():
                    if ndict[k] == True:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail = ';'.join(str(x) for x in tempk).replace(',', ':')
                nc.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                if j < 2:
                    continue
                param_name = ""
                if params == 1:
                    param_name = "averaging:3:3"
                if params == 2:
                    param_name = "averaging:4:4"
                if params == 3:
                    param_name = "averaging:5:5"
                if params == 4:
                    param_name = "GaussianBlur:3:3"
                if params == 5:
                    param_name = "GaussianBlur:5:5"
                if params == 6:
                    param_name = "GaussianBlur:7:7"
                if params == 7:
                    param_name = "medianBlur:3"
                if params == 8:
                    param_name = "medianBlur:5"
                if params == 9:
                    param_name = "averaging:6:6"
                if params == 10:
                    param_name = "bilateralFilter:9:75:75"
                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('blur')
                csvrecord.append(param_name)
                csvrecord.append(param_name)
                csvrecord.append(threshold)

                csvrecord.append(new_covered1)
                csvrecord.append(new_total1)
                csvrecord.append(covered_detail)

                csvrecord.append(yhat[0][0])
                csvrecord.append(label1[j][1])
                print(csvrecord)
                writer.writerow(csvrecord)
        print("all done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    parser.add_argument('--index', type=int, default=0,
                        help='different indice mapped to different transformations and params')
    args = parser.parse_args()
    epoch_testgen_coverage(args.index, args.dataset)
