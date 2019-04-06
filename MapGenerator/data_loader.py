import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
class DataLoader():
    def __init__(self, img_res=(128, 128)):
       # self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False,datapath ='', nImages=0):
        data_type = "train" if not is_testing else "test"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = list()
        pics = os.listdir(datapath)
        if nImages >0 and nImages <= len(pics):
            for file in np.random.choice(pics,size = 1000):
                path.append(datapath + file)
        else:
             for file in pics:
                path.append(datapath + file)        ##print(path)
        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, _w:, :], img[:,:_w, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)
            # If training => do random flip (independent random lr and ud flips)
            if not is_testing and np.random.random() < 0.5:
                img_A = np.flipud(img_A)
                img_B = np.flipud(img_B)
            
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False, datapath = '',nImages = 0):
        path = list()
        if nImages >0:
            for file in np.random.choice(os.listdir(datapath),size = 1000):
                path.append(datapath + file)
        else:
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
                half_w = int(w/2)
                img_A = img[:, half_w:, :]
                img_B = img[:, :half_w, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
