import glob
import argparse
import numpy as np
from collections import deque
from keras.models import load_model
from keras.models import Model as Kmodel
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from scipy import misc
from scipy.misc import imread, imresize, imsave
import sys
import os
from ncoverage import NCoverage
import csv
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imshow

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
        print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)

        self.threshold = 0.2
        #self.nc = NCoverage(self.model,self.threshold)
        s1 = self.model.get_layer('sequential_1')
        self.nc1 = NCoverage(s1,self.threshold)
        #print(s1.summary())
        

        s2 = self.model.get_layer('sequential_2')
        #print(s2.summary())
        self.nc2 = NCoverage(s2,self.threshold)
        
        
        s3 = self.model.get_layer('sequential_3')
        #print(s3.summary())
        self.nc3 = NCoverage(s3,self.threshold)
        
        
        i1 = self.model.get_layer('input_1')

        self.i1_model = Kmodel(input = self.model.inputs, output = i1.output)


        
    def predict(self, img):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return 0, 0, self.mean_angle[0],0,0,0,0,0,0

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return 0, 0, self.mean_angle[0],0,0,0,0,0,0

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8) # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:,:,::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
           
            
            #print(self.model.summary())
            #for layer in self.model.layers:
                #print (layer.name)
            
            i1_outputs = self.i1_model.predict(X)
            '''
            layerlist1 = self.nc1.update_coverage(i1_outputs)
            covered_neurons1, total_neurons1, p = self.nc1.curr_neuron_cov()
            c1 = covered_neurons1
            t1 = total_neurons1
            
            layerlist2 = self.nc2.update_coverage(i1_outputs)
            covered_neurons2, total_neurons2, p = self.nc2.curr_neuron_cov()
            c2 = covered_neurons2
            t2 = total_neurons2
            
            layerlist3 = self.nc3.update_coverage(i1_outputs)
            covered_neurons3, total_neurons3, p = self.nc3.curr_neuron_cov()
            c3 = covered_neurons3
            t3 = total_neurons3
            covered_neurons = covered_neurons1 + covered_neurons2 + covered_neurons3
            total_neurons  = total_neurons1 + total_neurons2 + total_neurons3
            '''
            rs1 = self.s1_model.predict(i1_outputs)
            rs2 = self.s2_model.predict(i1_outputs)
            rs3 = self.s3_model.predict(i1_outputs)
            #return covered_neurons, total_neurons, self.model.predict(X)[0][0],c1,t1,c2,t2,c3,t3
            return 0, 0, self.model.predict(X)[0][0],rs1[0][0],rs2[0][0],rs3[0][0],0,0,0
    
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

            X = X[:,:,::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
           
            
            #print(self.model.summary())
            #for layer in self.model.layers:
                #print (layer.name)
            
            i1_outputs = self.i1_model.predict(X)
            
            d1 = self.nc1.update_coverage(i1_outputs)
            covered_neurons1, total_neurons1, p = self.nc1.curr_neuron_cov()
            c1 = covered_neurons1
            t1 = total_neurons1
            
            d2 = self.nc2.update_coverage(i1_outputs)
            covered_neurons2, total_neurons2, p = self.nc2.curr_neuron_cov()
            c2 = covered_neurons2
            t2 = total_neurons2
            
            d3 = self.nc3.update_coverage(i1_outputs)
            covered_neurons3, total_neurons3, p = self.nc3.curr_neuron_cov()
            c3 = covered_neurons3
            t3 = total_neurons3
            covered_neurons = covered_neurons1 + covered_neurons2 + covered_neurons3
            total_neurons  = total_neurons1 + total_neurons2 + total_neurons3

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
        print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)
        self.threshold = 0.2

def image_translation(img, params):
    
    rows,cols,ch = img.shape

    M = np.float32([[1,0,params[0]],[0,1,params[1]]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_scale(img, params):

    res = cv2.resize(img,None,fx=params[0], fy=params[1], interpolation = cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
    rows,cols,ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1,factor,0],[0,1,0]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def image_rotation(img, params):
    rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),params,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
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
        blur = cv2.blur(img,(3,3))
    if params == 2:
        blur = cv2.blur(img,(4,4))
    if params == 3:
        blur = cv2.blur(img,(5,5))
    if params == 4:
        blur = cv2.GaussianBlur(img,(3,3),0)
    if params == 5:
        blur = cv2.GaussianBlur(img,(4,4),0)
    if params == 6:
        blur = cv2.GaussianBlur(img,(5,5),0)
    if params == 7:
        blur = cv2.medianBlur(img,3)
    if params == 8:
        blur = cv2.medianBlur(img,4)
    if params == 9:
        blur = cv2.medianBlur(img,5)
    if params == 10:
        blur = cv2.bilateralFilter(img,9,75,75)
    return blur        



def rambo_testgen_coverage(dataset_path):
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    
    model = Model("./final_model.hdf5", "./X_train_mean.npy")
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

    
    with open('result/rambo_rq2_70000_images.csv', 'wb',0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index', 'image', 'tranformation', 'param_name',
                         'param_value','threshold','covered_neurons', 'total_neurons',
                         's1_covered', 's1_total','s1_detail',
                         's2_covered', 's2_total','s2_detail',
                         's3_covered', 's3_total','s3_detail',
                         'y_hat','label'])
            
        
        #seed input
        input_images = xrange(1, 1001)
        for i in input_images:
            j = i * 5
            csvrecord = []

            seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
            new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)

            seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
            new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)

            seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
            new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)
            filename, ext = os.path.splitext(str(filelist1[j]))
            
            if label1[j][0] != filename:
                print(filename + " not found in the label file")
                continue            
          
            

            tempk = []
            for k in d1.keys():
                if d1[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            #covered_detail1 = ';'.join(str(x) for x in tempk).replace(',', ':')
            covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')
            tempk = []
            for k in d2.keys():
                if d2[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

            tempk = []
            for k in d3.keys():
                if d3[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')       

            csvrecord.append(j-2)
            csvrecord.append(str(filelist1[j]))
            csvrecord.append('-')
            csvrecord.append('-')
            csvrecord.append('-')
            csvrecord.append(model.threshold)

            csvrecord.append(new_covered)
            csvrecord.append(new_total)
            csvrecord.append(c1)
            csvrecord.append(t1)
            csvrecord.append(covered_detail1)
            csvrecord.append(c2)
            csvrecord.append(t2)
            csvrecord.append(covered_detail2)
            csvrecord.append(c3)
            csvrecord.append(t3)
            csvrecord.append(covered_detail3)


            csvrecord.append(result)
            csvrecord.append(label1[j][1])
            print(csvrecord)
            writer.writerow(csvrecord)
            model.hard_reset()

        print("seed input done")
        
        #Translation   
    
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = [p*10, p*10]
            for i in input_images:
                j = i * 5
                csvrecord = []

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_translation,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_translation,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_translation,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                
                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')        

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('translation')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()

        print("translation done")
        
        #Scale
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = [p*0.5+1, p*0.5+1]            

            for i in input_images:
                j = i * 5
                csvrecord = []
                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))
                seed_image = image_scale(seed_image,map(lambda x:x+0, params))      
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))
                seed_image = image_scale(seed_image,map(lambda x:x+0.1, params))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                seed_image = image_scale(seed_image,map(lambda x:x+0.2, params))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,None,None)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')




                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('scale')
                csvrecord.append('x:y')
                csvrecord.append(':'.join(str(x) for x in params))
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()

        print("scale done")

        
        #Shear
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = 0.1*p  
            for i in input_images:
                j = i * 5
                csvrecord = []
                
                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_shear,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_shear,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_shear,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('shear')
                csvrecord.append('factor')
                csvrecord.append(params)
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()
        print("sheer done")

        
        #Rotation
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = p*3
            for i in input_images:
                j = i * 5
                csvrecord = []

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_rotation,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_rotation,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_rotation,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('rotation')
                csvrecord.append('angle')
                csvrecord.append(params)
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()
        
        print("rotation done")
        

        #Contrast
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = 1 + p*0.2 
            for i in input_images:
                j = i * 5
                csvrecord = []
                  
                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_contrast,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_contrast,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_contrast,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('contrast')
                csvrecord.append('gain')
                csvrecord.append(params)
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()
        
        print("contrast done")


        
        #Brightness
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = p * 10 
            for i in input_images:
                j = i * 5
                csvrecord = []
                   
                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_brightness,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_brightness,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_brightness,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('brightness')
                csvrecord.append('bias')
                csvrecord.append(params)
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()
        print("brightness done")
        
        
        #blur
        input_images = xrange(1, 1001)
        for p in xrange(1, 11):
            params = p 
            for i in input_images:
                j = i * 5
                csvrecord = []

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-2]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_blur,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j-1]))        
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_blur,params)

                seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                new_covered, new_total, result,c1,t1,d1,c2,t2,d2,c3,t3,d3 = model.predict1(seed_image,image_blur,params)
                filename, ext = os.path.splitext(str(filelist1[j]))
                

                if label1[j][0] != filename:
                    print(filename + " not found in the label file")
                    continue            
              
                

                tempk = []
                for k in d1.keys():
                    if d1[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d2.keys():
                    if d2[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

                tempk = []
                for k in d3.keys():
                    if d3[k]:
                        tempk.append(k)
                tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
                covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')


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
                    param_name = "GaussianBlur:4:4"
                if params == 6:
                    param_name = "GaussianBlur:5:5"
                if params == 7:
                    param_name = "medianBlur:3"
                if params == 8:
                    param_name = "medianBlur:4"
                if params == 9:
                    param_name = "medianBlur:5"
                if params == 10:
                    param_name = "bilateralFilter:9:75:75"
                
                csvrecord.append(j-2)
                csvrecord.append(str(filelist1[j]))
                csvrecord.append('blur')
                csvrecord.append(param_name)
                csvrecord.append('-')
                csvrecord.append(model.threshold)

                csvrecord.append(new_covered)
                csvrecord.append(new_total)
                csvrecord.append(c1)
                csvrecord.append(t1)
                csvrecord.append(covered_detail1)
                csvrecord.append(c2)
                csvrecord.append(t2)
                csvrecord.append(covered_detail2)
                csvrecord.append(c3)
                csvrecord.append(t3)
                csvrecord.append(covered_detail3)


                csvrecord.append(result)
                csvrecord.append(label1[j][1])
                #print(csvrecord)
                print(csvrecord[:5])
                writer.writerow(csvrecord)
                model.hard_reset()
        print("all done")



if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    args = parser.parse_args()
    rambo_testgen_coverage(args.dataset)