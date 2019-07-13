import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
class DataLoader():
    def __init__(self, inpt_img_res=(128, 128),opt_img_res=(128, 128)):
       # self.dataset_name = dataset_name
        self.inpt_img_res = inpt_img_res
        self.opt_img_res=opt_img_res

    def form_images(img):
        return (0)## function to be built if tests are sucessful

    def load_data(self, batch_size=1, is_testing=False,datapath =''):
        data_type = "train" if not is_testing else "test"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = list()
        for file in os.listdir(datapath):
            path.append(datapath + file)
        ##print(path)
        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, _w:, :], img[:,:_w, :]
            ## now append filter blocked element of B onto A as "known" map element
            blocking = np.zeros(np.prod(img_A.shape))
            blocking = np.reshape(blocking, img_A.shape)
            xmin = np.random.randint(0,img_A.shape[0])
            xmax = np.random.randint(xmin,img_A.shape[0])
            ymin = np.random.randint(0,img_A.shape[1])
            ymax = np.random.randint(ymin,img_A.shape[1])
            blocking[xmin:xmax,ymin:ymax,:] = 1

               # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            blocked_A  = np.multiply (img_A , blocking)
            img_B = np.concatenate((img_B,blocked_A),axis = 1)
            img_A = scipy.misc.imresize(img_A, self.opt_img_res)
            img_B = scipy.misc.imresize(img_B, self.inpt_img_res)

         

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False, datapath = ''):
        path = list()
        for file in os.listdir(datapath):
            path.append(datapath + file)
        #print(path)
        data_type = "train" if not is_testing else "val"
     #   path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                _w = int(w/2)
                img_A, img_B = img[:, _w:, :], img[:,:_w, :]
                ## now append filter blocked element of B onto A as "known" map element
                blocking = np.zeros(np.prod(img_A.shape))
                blocking = np.reshape(blocking, img_A.shape)
                xmin = np.random.randint(0,img_A.shape[0])
                xmax = np.random.randint(xmin,img_A.shape[0])
                ymin = np.random.randint(0,img_A.shape[1])
                ymax = np.random.randint(ymin,img_A.shape[1])
                blocking[xmin:xmax,ymin:ymax,:] = 1

               # If training => do random flip
                if not is_testing and np.random.random() < 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                blocked_A  = np.multiply (img_A , blocking)
                img_B = np.concatenate((img_B,blocked_A),axis = 1)
                img_A = scipy.misc.imresize(img_A, self.opt_img_res)
                img_B = scipy.misc.imresize(img_B, self.inpt_img_res)
         

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
