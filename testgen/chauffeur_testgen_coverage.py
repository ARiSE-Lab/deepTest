from __future__ import print_function
from keras import backend as K
from keras import metrics
from keras.models import model_from_json
import argparse
from collections import deque
from scipy import misc
import csv
import os
import sys
from ncoverage import NCoverage
import cv2
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')
# keras 1.2.2 tf:1.2.0
class ChauffeurModel(object):
    def __init__(self,
                 cnn_json_path,
                 cnn_weights_path,
                 lstm_json_path,
                 lstm_weights_path, only_layer=""):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 100

        self.threshold_cnn = 0.1
        self.threshold_lstm = 0.4
        self.timestepped_x = np.empty((1, self.timesteps, 8960))
        self.nc_lstm = NCoverage(self.lstm, self.threshold_lstm)
        self.nc_encoder = NCoverage(self.encoder, self.threshold_cnn,
                                    exclude_layer=['pool', 'fc', 'flatten'],
                                    only_layer=only_layer)
        self.steps = deque()
        #print(self.lstm.summary())
        #self.nc = NCoverage(self.lstm,self.threshold)

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img-(255.0/2))/255.0)

            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    #def make_stateful_predictor(self):
        #steps = deque()

    def predict_fn(self, img, dummy=2):
        # preprocess image to be YUV 320x120 and equalize Y histogram
        steps = self.steps
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img[120:240, :, :]
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = ((img-(255.0/2))/255.0)
        img1 = img
        # apply feature extractor
        img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

        # initial fill of timesteps
        if not len(steps):
            for _ in xrange(self.timesteps):
                steps.append(img)

        # put most recent features at end
        steps.popleft()
        steps.append(img)
        #print(len(steps))
        #timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
        if dummy == 0:
            return 0, 0, 0, 0, 0, 0, 0
        for i, img in enumerate(steps):
            self.timestepped_x[0, i] = img

        '''
        self.nc.update_coverage(timestepped_x)
        covered_neurons, total_neurons, p = self.nc.curr_neuron_cov()
        print('input covered {} neurons'.format(covered_neurons))
        print('total {} neurons'.format(total_neurons))
        print('percentage {}'.format(p))
        '''
        cnn_ndict = self.nc_encoder.update_coverage(img1.reshape((1, 120, 320, 3)))
        cnn_covered_neurons, cnn_total_neurons, p = self.nc_encoder.curr_neuron_cov()
        if dummy == 1:
            return cnn_ndict, cnn_covered_neurons, cnn_total_neurons, 0, 0, 0, 0
        lstm_ndict = self.nc_lstm.update_coverage(self.timestepped_x)
        lstm_covered_neurons, lstm_total_neurons, p = self.nc_lstm.curr_neuron_cov()
        return cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict,\
        lstm_covered_neurons, lstm_total_neurons,\
        self.lstm.predict_on_batch(self.timestepped_x)[0, 0] / self.scale

    #return predict_fn

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

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

metrics.rmse = rmse

def chauffeur_testgen_coverage(index, dataset_path):
    cnn_json_path = "./cnn.json"
    cnn_weights_path = "./cnn.weights"
    lstm_json_path = "./lstm.json"
    lstm_weights_path = "./lstm.weights"
    K.set_learning_phase(0)
    model = ChauffeurModel(
        cnn_json_path,
        cnn_weights_path,
        lstm_json_path,
        lstm_weights_path)

    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")


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

    index = int(index)

    #seed inputs
    with open('result/chauffeur_rq2_70000_images.csv', 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'image', 'tranformation', 'param_name', 'param_value',
                         'cnn_threshold', 'cnn_covered_neurons', 'cnn_total_neurons',
                         'cnn_covered_detail', 'lstm_threshold', 'lstm_covered_neurons',
                         'lstm_total_neurons', 'lstm_covered_detail', 'y_hat', 'label'])
        if index/2 == 0:
            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)
            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, \
                lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)


                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()
                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue


                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append('-')
                csvrecord.append(model.threshold_cnn)


                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)



                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)

            print("seed input done")

        #Translation

        if index/2 >= 1 and index/2 <= 10:
            #for p in xrange(1,11):
            p = index/2
            params = [p*10, p*10]

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_translation(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_translation(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_translation(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)


                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()
                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue




                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('translation')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.threshold_cnn)



                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)



                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("translation done")

        #Scale
        if index/2 >= 11 and index/2 <= 20:
        #for p in xrange(1,11):
            p = index/2-10
            params = [p*0.5+1, p*0.5+1]
            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_scale(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_scale(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_scale(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)


                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue


                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('scale')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.threshold_cnn)



                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)



                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)



        print("scale done")


        #Shear 42-61
        if index/2 >= 21 and index/2 <= 30:
        #for p in xrange(1,11):
            p = index/2-20
        #for p in xrange(1,11):
            params = 0.1*p

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_shear(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_shear(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_shear(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)


                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('shear')
                csvrecord.append('factor')
                csvrecord.append(params)
                csvrecord.append(model.threshold_cnn)

                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)

                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("sheer done")

        #Rotation 62-81
        if index/2 >= 31 and index/2 <= 40:
            p = index/2-30
        #for p in xrange(1,11):
            params = p*3

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_rotation(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_rotation(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_rotation(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)

                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('rotation')
                csvrecord.append('angle')
                csvrecord.append(params)
                csvrecord.append(model.threshold_cnn)


                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)

                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)


        print("rotation done")

        #Contrast 82-101
        if index/2 >= 41 and index/2 <= 50:
            p = index/2-40
        #or p in xrange(1,11):
            params = 1 + p*0.2

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_contrast(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_contrast(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_contrast(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)

                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('contrast')
                csvrecord.append('gain')
                csvrecord.append(params)
                csvrecord.append(model.threshold_cnn)

                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)

                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("contrast done")


        #Brightness 102-121
        if index/2 >= 51 and index/2 <= 60:
            p = index/2-50
        #for p in xrange(1,11):
            params = p * 10

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_brightness(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_brightness(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)

            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_brightness(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)


                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))


                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('brightness')
                csvrecord.append('bias')
                csvrecord.append(params)
                csvrecord.append(model.threshold_cnn)

                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)

                csvrecord.append(yhat)
                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)


        print("brightness done")

        #blur 122-141
        if index/2 >= 61 and index/2 <= 70:
            p = index/2-60
        #for p in xrange(1,11):
            params = p

            if index%2 == 0:
                for j in xrange(2000, 2100):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_blur(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)
            else:
                for j in xrange(2500, 2600):
                    seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                    seed_image = image_blur(seed_image, params)
                    model.predict_fn(seed_image, dummy=0)

            if index%2 == 0:
                input_images = xrange(2100, 2600)
            else:
                input_images = xrange(2600, 3100)


            for i in input_images:
                j = i * 1
                csvrecord = []
                seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_blur(seed_image, params)

                cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)

                tempk = []
                for k in cnn_ndict.keys():
                    if cnn_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


                tempk = []
                for k in lstm_ndict.keys():
                    if lstm_ndict[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                model.nc_encoder.reset_cov_dict()

                model.nc_lstm.reset_cov_dict()
                #print(covered_neurons)
                #covered_neurons = nc.get_neuron_coverage(test_x)
                #print('input covered {} neurons'.format(covered_neurons))
                #print('total {} neurons'.format(total_neurons))

                filename, ext = os.path.splitext(str(filelist1[j]))
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
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
                csvrecord.append('-')
                csvrecord.append(model.threshold_cnn)


                csvrecord.append(cnn_covered_neurons)
                csvrecord.append(cnn_total_neurons)
                csvrecord.append(cnn_covered_detail)
                csvrecord.append(model.threshold_lstm)

                csvrecord.append(lstm_covered_neurons)
                csvrecord.append(lstm_total_neurons)
                csvrecord.append(lstm_covered_detail)



                csvrecord.append(yhat)

                csvrecord.append(label1[j][1])
                print(csvrecord[:8])
                writer.writerow(csvrecord)

        print("all done")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    parser.add_argument('--index', type=int, default=0,
                        help='different indice mapped to different transformations and params')
    args = parser.parse_args()
    chauffeur_testgen_coverage(args.index, args.dataset)
