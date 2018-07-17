'''
Generate images and steering angles from hmb3.bag
Modified from
https://github.com/udacity/self-driving-car/blob/master/steering-models/evaluation/generator.py
'''
import argparse
import rosbag
from StringIO import StringIO
from scipy import misc
import numpy as np
import csv

KEY_NAME = {
    '/vehicle/steering_report': 'steering',
    '/center_camera/image_color/c': 'image',
}

def update(msg, d):
    key = KEY_NAME.get(msg[0])
    if key is None: return
    d[key] = msg

def gen(bag):
    print 'Getting bag'
    bag = rosbag.Bag(bag)
    print 'Got bag'
    
    image = {}
    total = bag.get_message_count()
    count = 0
    for e in bag.read_messages():

        count += 1
        if count % 10000 == 0:
            print count, '/', total
        if e[0] in ['/center_camera/image_color/compressed']:
            #print(e)
            if len({'steering'} - set(image.keys())):
                continue
            if image['steering'][1].speed < 5.: continue
            s = StringIO(e[1].data)
            img = misc.imread(s)
            yield img, np.copy(img), image['steering'][1].speed,\
                  image['steering'][1].steering_wheel_angle, e[2].to_nsec()
            last_ts = e[2].to_nsec()
        else:
            update(e, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate images from hmb3')
    parser.add_argument('--bagfile', type=str, help='Path to ROS bag')
    args = parser.parse_args()
    data_iter = gen(args.bagfile)
    with open('hmb3/hmb3_steering.csv', 'wb',0) as hmb3csv:
        writer = csv.writer(hmb3csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['timestamp','steering angle'])
        
        for image_pred, image_disp, speed, steering, ts in data_iter:
            misc.imsave('hmb3/'+ str(ts) + '.jpg', image_disp)
            #print(ts)
            csvcontent = []
            csvcontent.append(ts)
            csvcontent.append(steering)
            writer.writerow(csvcontent)
