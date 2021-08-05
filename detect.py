from __future__ import division
import time
import torch 
from torch.autograd import Variable
import cv2 
from utils import *
from darknet import *
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import random


class DetectYoLov3:
    def __init__(self, cfg='cfg/yolov3.cfg', weights='weights/yolov3.weights'):
        self.num_classes = 80
        self.classes = load_classes("coco.names")
        self.batch_size = 1
        self.confidence = float(0.5)
        self.nms_thesh = float(0.4)
        self.CUDA = torch.cuda.is_available()

        print("Loading network.....")
        self.model = Darknet(cfg)
        self.model.load_weights(weights)
        print("Network successfully loaded")

        self.model.net_info["height"] = 416
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()

        self.model.eval()

    def write(self, x, results):
        cls = int(x[-1])
        img = results[int(x[0])]
        if cls == 15:
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            color = random.choice(self.colors)
            label = "{0}".format(self.classes[cls])
            cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]),int(c2[1])), color, 3)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]),int(c2[1])), color, -1)
            cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img

    def read_list_images(self):
        try:
            self.imlist = [osp.join(osp.realpath('.'), 'image', img) for img in os.listdir('image')]
        except NotADirectoryError:
            self.imlist = []
            self.imlist.append(osp.join(osp.realpath('.'), 'image'))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format('image'))
            exit()

        if not os.path.exists('det_list_images'):
            os.makedirs('det_list_images')

        self.loaded_ims = [cv2.imread(x) for x in self.imlist]
        #PyTorch Variables for images
        self.im_batches = list(map(prep_image, self.loaded_ims, [self.inp_dim for x in range(len(self.imlist))]))

    def read_image(self, photo):
        if not os.path.exists('det_image'):
            os.makedirs('det_image')
        self.loaded_ims = [cv2.imread(photo)]
        #PyTorch Variables for images
        self.im_batches = list(map(prep_image, self.loaded_ims,[self.inp_dim]))

    def detect_images(self):
        im_dim_list = [(x.shape[1], x.shape[0]) for x in self.loaded_ims]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        if self.CUDA:
            im_dim_list = im_dim_list.cuda()

        leftover = 0
        if (len(im_dim_list) % self.batch_size):
            leftover = 1

        if self.batch_size != 1:
            num_batches = len(self.imlist) // self.batch_size + leftover            
            self.im_batches = [torch.cat((self.im_batches[i*self.batch_size : min((i +  1)*self.batch_size,
                                len(self.im_batches))]))  for i in range(num_batches)]

        write = 0
        start_det_loop = time.time()
        for i, batch in enumerate(self.im_batches):
            if self.CUDA:
                batch = batch.cuda()

            with torch.no_grad():
                prediction = self.model(Variable(batch), self.CUDA)

            prediction = write_results(prediction,
                                       self.confidence,
                                       self.num_classes,
                                       nms_conf = self.nms_thesh)

            end = time.time()

            prediction[:,0] += i*self.batch_size

            if not write:
                output = prediction  
                write = 1
            else:
                output = torch.cat((output,prediction))

            if self.CUDA:
                torch.cuda.synchronize()
        try:
            output
        except NameError:
            print ("No detections were made")
            exit()
        self.box_in_real_image(im_dim_list, output, start_det_loop)

    def box_in_real_image(self, im_dim_list, output, start_det_loop):
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

        scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)

        output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        output_recast = time.time()
        self.colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: self.write(x, self.loaded_ims), output))
        try:
            for i, x in enumerate(self.imlist):
                det_name = 'det_list_images/det_{}'.format(x.split('\\')[-1])
                cv2.imwrite(det_name, self.loaded_ims[i])
            print("SUMMARY")
            print("----------------------------------------------------------")
            print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
            print("{:25s}: {:2.3f}".format("Detection (" + str(len(self.imlist)) +  " images)", output_recast - start_det_loop))
            print("----------------------------------------------------------")
        except Exception:
            cv2.imwrite('det_image/detected.jpg', self.loaded_ims[0])
            print("SUMMARY")
            print("----------------------------------------------------------")
            print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
            print("{:25s}: {:2.3f}".format("Detection (1 images)", output_recast - start_det_loop))
            print("----------------------------------------------------------")


        torch.cuda.empty_cache()

if __name__ == '__main__':
    detect = DetectYoLov3()
    detect.read_image('image/100.jpg')
    detect.detect_images()
