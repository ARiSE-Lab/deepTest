'''
Leverage neuron coverage to guide the generation of images from combinations of transformations.
'''
from __future__ import print_function
import argparse
import numpy as np
import os
import sys
from collections import defaultdict
import random
from keras import backend as K
from ncoverage import NCoverage
import csv
import cv2
import pickle
from keras import backend as K
from keras.models import model_from_json
import argparse
from collections import deque


reload(sys)
sys.setdefaultencoding('utf8')
# keras 1.2.2 tf:1.2.0
class ChauffeurModel(object):
    '''
    Chauffeur model with integrated neuron coverage
    '''
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
        self.nc_encoder = NCoverage(self.encoder, self.threshold_cnn, exclude_layer=['pool', 'fc', 'flatten'], only_layer=only_layer)
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

    def predict_fn(self, img, test=0):
        # test == 0: update the coverage only
        # test == 1: test if the input image will increase the current coverage
        steps = self.steps
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img[120:240, :, :]
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = ((img-(255.0/2))/255.0)
        img1 = img

        if test == 1:
            return self.nc_encoder.is_testcase_increase_coverage(img1.reshape((1, 120, 320, 3)))
        else:
            cnn_ndict = self.nc_encoder.update_coverage(img1.reshape((1, 120, 320, 3)))
            cnn_covered_neurons, cnn_total_neurons, cnn_p = self.nc_encoder.curr_neuron_cov()
            return cnn_covered_neurons, cnn_total_neurons, cnn_p

    #return predict_fn
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



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

    rows,cols,ch = img.shape
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

def chauffeur_guided(dataset_path):
    model_name = "cnn"
    image_size = (128, 128)
    threshold = 0.2

    root = ""
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")
    new_input = "./new/"
    # Model build
    # ---------------------------------------------------------------------------------
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


    filelist1 = []
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    newlist = []
    for file in sorted(os.listdir(new_input)):
        if file.endswith(".jpg"):
            newlist.append(file)
    flag = 0
    #flag:0 start from beginning
    #flag:1 initialize from pickle files

    '''
    Pickle files are used for continuing the search after rerunning the script.
    Delete all pkl files and generated images for starting from the beginnning.
    '''
    if os.path.isfile("chauffeur_covdict2.pkl") and \
                os.path.isfile("chauffeur_stack.pkl") and \
                os.path.isfile("chauffeur_queue.pkl") and \
                os.path.isfile("generated.pkl"):
        with open('chauffeur_covdict2.pkl', 'rb') as input:
            covdict = pickle.load(input)
        with open('chauffeur_stack.pkl', 'rb') as input:
            chauffeur_stack = pickle.load(input)
        with open('chauffeur_queue.pkl', 'rb') as input:
            chauffeur_queue = pickle.load(input)
        with open('generated.pkl', 'rb') as input:
            generated = pickle.load(input)
        flag = 1


    if flag == 0:
        filewrite = "wb"
        chauffeur_queue = deque()
        chauffeur_stack = []
        generated = 0
    else:
        model.nc_encoder.set_covdict(covdict)
        filewrite = "ab"
        print("initialize from files")

    C = 0 # covered neurons
    P = 0 # covered percentage
    T = 0 # total neurons
    transformations = [image_translation, image_scale, image_shear,
                       image_rotation, image_contrast, image_brightness2, image_blur]
    params = []
    params.append(list(xrange(-50, 50)))
    params.append(list(map(lambda x: x*0.1, list(xrange(5, 20)))))
    params.append(list(map(lambda x: x*0.1, list(xrange(-5, 5)))))
    params.append(list(xrange(-30, 30)))
    params.append(list(map(lambda x: x*0.1, list(xrange(1, 20)))))
    params.append(list(xrange(-21, 21)))
    params.append(list(xrange(1, 11)))

    maxtrynumber = 10
    cache = deque()
    #load nc, generation, population
    with open('result/chauffeur_rq3_100_2.csv', filewrite, 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if flag == 0:
            writer.writerow(['id', 'seed image(root)', 'parent image', 'generated images',
                             'total_covered','total_neurons', 'coverage_percentage'])
            #initialize population and coverage
            print("compute coverage of original population")
            input_images = xrange(0, 100)
            for i in input_images:
                j = 2100 + i * 10
                chauffeur_queue.append(os.path.join(seed_inputs1, filelist1[j]))
        #exitcount = 0
        while len(chauffeur_queue) > 0:
            current_seed_image = chauffeur_queue[0]
            print(str(len(chauffeur_queue)) + " images are left.")
            if len(chauffeur_stack) == 0:
                chauffeur_stack.append(current_seed_image)

            while len(chauffeur_stack) > 0:
                try:
                    image_file = chauffeur_stack[-1]
                    print("current image in stack " + image_file)
                    image = cv2.imread(image_file)
                    covered, total, p = model.predict_fn(image)
                    new_generated = False
                    for i in xrange(maxtrynumber):

                        #exitcount = exitcount + 1
                        tid = random.sample([0,1,2,3,4,5,6], 2)
                        if len(cache) > 0:
                            tid[0] = cache.popleft()
                        for j in xrange(2):
                            new_image = image
                            transformation = transformations[tid[j]]
                            #random choose parameter
                            param = random.sample(params[tid[j]], 1)
                            param = param[0]
                            print("transformation " + str(transformation) + "  parameter " + str(param))
                            new_image = transformation(new_image,param)

                        if model.predict_fn(new_image, test = 1):
                            print("Generated image increases coverage and will be added to population.")
                            cache.append(tid[0])
                            cache.append(tid[1])
                            generated = generated + 1
                            name = os.path.basename(current_seed_image)+'_' + str(generated)+'.jpg'
                            name = os.path.join(new_input,name)
                            cv2.imwrite(name,new_image)
                            chauffeur_stack.append(name)

                            covered, total, p = model.predict_fn(image)
                            C = covered
                            T = total
                            P = p
                            csvrecord = []
                            csvrecord.append(100-len(chauffeur_queue))
                            csvrecord.append(os.path.basename(current_seed_image))
                            if len(chauffeur_stack) >= 2:
                                parent = os.path.basename(chauffeur_stack[-2])
                            else:
                                parent = os.path.basename(current_seed_image)
                            csvrecord.append(parent)
                            csvrecord.append(generated)
                            csvrecord.append(C)
                            csvrecord.append(T)
                            csvrecord.append(P)
                            print(csvrecord)
                            writer.writerow(csvrecord)
                            new_generated = True
                            break
                        else:
                            print("Generated image does not increase coverage.")
                        '''
                        # If the memory is not enough, the following code can be used to exit.
                        # Re-runing the script will continue from previous progree.
                        print("exitcount: " + str(exitcount))
                        if exitcount % 30 == 0:
                            exit()
                        '''

                    if not new_generated:
                        chauffeur_stack.pop()

                    save_object(chauffeur_stack, 'chauffeur_stack.pkl')
                    save_object(chauffeur_queue, 'chauffeur_queue.pkl')
                    save_object(model.nc_encoder.cov_dict, 'chauffeur_covdict2.pkl')
                    save_object(generated, 'generated.pkl')
                    '''
                        # If the memory is not enough, the following code can be used to exit.
                        # Re-runing the script will continue from previous progree.
                        print("exitcount: " + str(exitcount))
                        if exitcount % 30 == 0:
                            exit()
                    '''
                    
                except:
                    print("value error")
                    chauffeur_stack.pop() 
                    save_object(chauffeur_stack, 'chauffeur_stack.pkl')
                    save_object(chauffeur_queue, 'chauffeur_queue.pkl')   

            chauffeur_queue.popleft()
            #maxtrynumber = maxtrynumber + 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    chauffeur_guided(args.dataset)
