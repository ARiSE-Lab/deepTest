from __future__ import print_function
import argparse
import sys
import os
import numpy as np
from collections import deque
from keras.models import load_model
from keras.models import Model as Kmodel
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
import random
import pickle
from scipy import misc
from ncoverage import NCoverage
import csv
import cv2
from PIL import Image
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

        self.threshold = 0.2
        #self.nc = NCoverage(self.model,self.threshold)
        s1 = self.model.get_layer('sequential_1')
        self.nc1 = NCoverage(s1, self.threshold)
        #print(s1.summary())


        s2 = self.model.get_layer('sequential_2')
        #print(s2.summary())
        self.nc2 = NCoverage(s2, self.threshold)


        s3 = self.model.get_layer('sequential_3')
        #print(s3.summary())
        self.nc3 = NCoverage(s3, self.threshold)


        i1 = self.model.get_layer('input_1')

        self.i1_model = Kmodel(input=self.model.inputs, output=i1.output)



    def predict(self, img):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle[0]

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle[0]

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
            return self.model.predict(X)[0][0]

    def predict1(self, img, transform, params):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return 0, 0, self.mean_angle[0],0,0,0,0,0,0,0,0,0

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return 0, 0, self.mean_angle[0],0,0,0,0,0,0,0,0,0

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)

            if transform != None and params != None:
                X = transform(X, params)

            X = X[:, :, ::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0


            #print(self.model.summary())
            #for layer in self.model.layers:
                #print (layer.name)

            i1_outputs = self.i1_model.predict(X)

            d1 = self.nc1.update_coverage(i1_outputs)
            covered_neurons1, total_neurons1, p1 = self.nc1.curr_neuron_cov()
            c1 = covered_neurons1
            t1 = total_neurons1

            d2 = self.nc2.update_coverage(i1_outputs)
            covered_neurons2, total_neurons2, p2 = self.nc2.curr_neuron_cov()
            c2 = covered_neurons2
            t2 = total_neurons2

            d3 = self.nc3.update_coverage(i1_outputs)
            covered_neurons3, total_neurons3, p3 = self.nc3.curr_neuron_cov()
            c3 = covered_neurons3
            t3 = total_neurons3
            covered_neurons = covered_neurons1 + covered_neurons2 + covered_neurons3
            total_neurons = total_neurons1 + total_neurons2 + total_neurons3

            return covered_neurons, total_neurons, self.model.predict(X)[0][0],c1,t1,d1,c2,t2,d2,c3,t3,d3
            #return 0, 0, self.model.predict(X)[0][0],rs1[0][0],rs2[0][0],rs3[0][0],0,0,0

    def hard_reset(self):

        self.mean_angle = np.array([-0.004179079])
        #print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)
        self.threshold = 0.2
        #self.nc.reset_cov_dict()
        self.nc1.reset_cov_dict()
        self.nc2.reset_cov_dict()
        self.nc3.reset_cov_dict()


    def soft_reset(self):

        self.mean_angle = np.array([-0.004179079])
        print (self.mean_angle)
        self.img0 = None
        self.state = deque(maxlen=2)
        self.threshold = 0.2

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
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation = cv2.INTER_CUBIC)
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
    b, g, r = cv2.split(img)
    b = cv2.add(b, beta)
    g = cv2.add(g, beta)
    r = cv2.add(r, beta)
    new_img = cv2.merge((b, g, r))
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
        blur = cv2.blur(img, (7, 7))
    return blur

def update_dict(dict1, covdict):
    r = False
    for k in covdict.keys():
        if covdict[k] and not dict1[k]:
            dict1[k] = True
            r = True
    return r

def is_update_dict(dict1, covdict):

    for k in covdict.keys():
        if covdict[k] and not dict1[k]:
            return True
    return False

def get_current_coverage(covdict):
    covered_neurons = len([v for v in covdict.values() if v])
    total_neurons = len(covdict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def rambo_guided(dataset_path):
    model_name = "cnn"
    image_size = (128, 128)
    threshold = 0.2

    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    new_input = "./new/"

    model = Model("./final_model.hdf5", "./X_train_mean.npy")

    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

    filelist1 = []
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    newlist = []
    newlist = [os.path.join(new_input, o) for o in os.listdir(new_input) if os.path.isdir(os.path.join(new_input, o))]

    dict1 = dict(model.nc1.cov_dict)
    dict2 = dict(model.nc2.cov_dict)
    dict3 = dict(model.nc3.cov_dict)

    flag = 0
    if os.path.isfile("rambo_stack.pkl") and os.path.isfile("rambo_queue.pkl") and os.path.isfile("generated.pkl") and os.path.isfile("covdict1.pkl") and os.path.isfile("covdict2.pkl") and os.path.isfile("covdict3.pkl"):
        with open('rambo_stack.pkl', 'rb') as input:
            rambo_stack = pickle.load(input)
        with open('rambo_queue.pkl', 'rb') as input:
            rambo_queue = pickle.load(input)
        with open('generated.pkl', 'rb') as input:
            generated = pickle.load(input)
        with open('covdict1.pkl', 'rb') as input:
            dict1 = pickle.load(input)
        with open('covdict2.pkl', 'rb') as input:
            dict2 = pickle.load(input)
        with open('covdict3.pkl', 'rb') as input:
            dict3 = pickle.load(input)
        flag = 1
    else:
        flag = 0

    if flag == 0:
        rambo_queue = deque()
        rambo_stack = []
        generated = 0
        filewrite = "wb"
    else:

        filewrite = "ab"
        print("initialize from files")

    C = 0 # covered neurons
    P = 0 # covered percentage
    T = 0 # total neurons
    maxtrynumber = 10
    cache = deque()
    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur]
    params = []
    params.append(list(xrange(-201, 201)))
    params.append(list(map(lambda x: x*0.1, list(xrange(5, 20)))))
    params.append(list(map(lambda x: x*0.1, list(xrange(-21, 21)))))
    params.append(list(xrange(-170, 170)))
    params.append(list(map(lambda x: x*0.1, list(xrange(1, 100)))))
    params.append(list(xrange(-101, 101)))
    params.append(list(xrange(1, 11)))
    

    with open('result/rambo_rq3_100_2_1.csv', filewrite, 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if flag == 0:
            writer.writerow(['id', 'seed image(root)', 'parent image', 'generated images',
                             'total_covered', 'total_neurons', 'coverage_percentage',
                             's1_covered', 's1_total', 's1_percentage',
                             's2_covered', 's2_total', 's2_percentage',
                             's3_covered', 's3_total', 's3_percentage'])
            #initialize population and coverage
            print("compute coverage of original population")
            input_images = xrange(1, 101)
            for i in input_images:
                j = i * 50
                image_file_group = []
                image_file_group.append(os.path.join(seed_inputs1, filelist1[j-2]))
                image_file_group.append(os.path.join(seed_inputs1, filelist1[j-1]))
                image_file_group.append(os.path.join(seed_inputs1, filelist1[j]))
                rambo_queue.append(image_file_group)
        #exitcount = 0
        #rambo_queue stores seed images group
        while len(rambo_queue) > 0:
            current_seed_image = rambo_queue[0]
            print(str(len(rambo_queue)) + " images are left.")
            if len(rambo_stack) == 0:
                rambo_stack.append(current_seed_image)

            #rambo_stack enable the depth first search
            while len(rambo_stack) > 0:
                try:
                    image_file_group = rambo_stack[-1]
                    image_group = []
                    print("current image in stack " + image_file_group[2])
                    seed_image1 = cv2.imread(image_file_group[0])
                    image_group.append(seed_image1)
                    seed_image2 = cv2.imread(image_file_group[1])
                    image_group.append(seed_image2)
                    seed_image3 = cv2.imread(image_file_group[2])
                    image_group.append(seed_image3)

                    model.predict1(seed_image1, None, None)
                    model.predict1(seed_image2, None, None)
                    new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image3, None, None)

                    #update cumulative coverage, dict1, dict2, dict3
                    update_dict(dict1, d1)
                    update_dict(dict2, d2)
                    update_dict(dict3, d3)
                    #get cumulative coverage
                    covered1, total1, p1 = get_current_coverage(dict1)
                    covered2, total2, p2 = get_current_coverage(dict2)
                    covered3, total3, p3 = get_current_coverage(dict3)
                    #reset model
                    model.hard_reset()

                    new_generated = False

                    for i in xrange(maxtrynumber):
                        tid = random.sample([0,1,2,3,4,5,6], 2)
                        new_image_group = []
                        params_group = []

                        #exitcount = exitcount + 1
                        if len(cache) > 0:
                            tid[0] = cache.popleft()
                        for j in xrange(2):

                            #random choose parameter for three images in a group. The parameters are slightly different.
                            param = random.sample(params[tid[j]], 1)
                            param = param[0]
                            param_id = params[tid[j]].index(param)
                            if param_id + 2 >= len(params[tid[j]]):
                                params_group.append([params[tid[j]][param_id-2], params[tid[j]][param_id-1], params[tid[j]][param_id]])
                            else:
                                params_group.append([params[tid[j]][param_id], params[tid[j]][param_id+1], params[tid[j]][param_id+2]])

                            transformation = transformations[tid[j]]
                            print("transformation " + str(transformation) + "  parameter " + str(param))


                        for k in xrange(3):
                            # transform all three images in a group
                            new_image = image_group[k]
                            for l in xrange(2):
                                transformation = transformations[tid[l]]
                                new_image = transformation(new_image, params_group[l][k])
                            new_image_group.append(new_image)

                        #Get coverage for this group
                        model.predict1(new_image_group[0], None, None)
                        model.predict1(new_image_group[1], None, None)
                        new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(new_image_group[2], None, None)

                        #check if some cumulative coverage is increased
                        b1 = is_update_dict(dict1, d1)
                        b2 = is_update_dict(dict2, d2)
                        b3 = is_update_dict(dict3, d3)
                        model.hard_reset()

                        new_image_file_group = []

                        if b1 or b2 or b3:
                            # if the coverage is increased, write these three images to files, 
                            # add the name of the new group to stack.
                            print("Generated image group increases coverage and will be added to population.")
                            cache.append(tid[0])
                            cache.append(tid[1])
                            new_generated = True
                            generated = generated + 1
                            foldername = str(generated)
                            folder = os.path.join(new_input, foldername)
                            if not os.path.exists(folder):
                                os.makedirs(folder)

                            for j in xrange(3):
                                filename = str(j)+'.jpg'
                                name = os.path.join(folder, filename)
                                new_image_file_group.append(name)
                                cv2.imwrite(name, new_image_group[j])

                            rambo_stack.append(new_image_file_group)


                            model.predict1(new_image_group[0], None, None)
                            model.predict1(new_image_group[1], None, None)
                            new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(new_image_group[2], None, None)
                            #update cumulative coverage
                            update_dict(dict1, d1)
                            update_dict(dict2, d2)
                            update_dict(dict3, d3)
                            #get cumulative coverage for output
                            covered1, total1, p1 = get_current_coverage(dict1)
                            covered2, total2, p2 = get_current_coverage(dict2)
                            covered3, total3, p3 = get_current_coverage(dict3)
                            model.hard_reset()
                            C = covered1 + covered2 + covered3
                            T = total1 + total2 + total3
                            P = C / float(T)

                            csvrecord = []
                            csvrecord.append(100-len(rambo_queue))
                            csvrecord.append(current_seed_image[2])
                            if len(rambo_stack) >= 2:
                                parent = rambo_stack[-2][2]
                            else:
                                parent = current_seed_image[2]
                            csvrecord.append(parent)
                            csvrecord.append(generated)
                            csvrecord.append(C)
                            csvrecord.append(T)
                            csvrecord.append(P)
                            csvrecord.append(covered1)
                            csvrecord.append(total1)
                            csvrecord.append(p1)
                            csvrecord.append(covered2)
                            csvrecord.append(total2)
                            csvrecord.append(p2)
                            csvrecord.append(covered3)
                            csvrecord.append(total3)
                            csvrecord.append(p3)
                            print(csvrecord)
                            writer.writerow(csvrecord)
                            save_object(generated, 'generated.pkl')
                            save_object(rambo_stack, 'rambo_stack.pkl')
                            save_object(rambo_queue, 'rambo_queue.pkl')
                            save_object(dict1, 'covdict1.pkl')
                            save_object(dict2, 'covdict2.pkl')
                            save_object(dict3, 'covdict3.pkl')
                            '''
                            # If the memory is not enough, the following code can be used to exit.
                            # Re-runing the script will continue from previous progree.
                            if generated % 100 == 0 or exitcount % 200 == 0:
                                exit()
                            '''                            
                            break
                        else:
                            print("Generated image group does not increase coverage.")
                        '''
                        # If the memory is not enough, the following code can be used to exit.
                        # Re-runing the script will continue from previous progree.
                        if generated % 100 == 0 or exitcount % 100 == 0:
                            exit()
                        '''
                    if not new_generated:
                        rambo_stack.pop()
                        save_object(rambo_stack, 'rambo_stack.pkl')
                        save_object(rambo_queue, 'rambo_queue.pkl')
                except:
                    print("value error")
                    rambo_stack.pop()
                    save_object(rambo_stack, 'rambo_stack.pkl')
                    save_object(rambo_queue, 'rambo_queue.pkl')
            rambo_queue.popleft()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    rambo_guided(args.dataset)
