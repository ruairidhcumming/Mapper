
import os
import numpy as np
from keras.utils import plot_model
import pix3pix
from  data_loader import DataLoader
### picture directories as str
picspath = str('C:\\Users\\ruair\\Documents\\rolerball\\screenshotOverhead\\')


print('creating Pix3Pix model')

#create gan model 
model = pix3pix.Pix2Pix()
## plot_model(model, to_file = 'C:\\Users\\ruair\\Documents\\rolerball')
## that was easy, pix2pix and dataloader modified from Keras-GAN and keras-contrib respectively

model.train(epochs = 1 , batch_size = 2, datapath = picspath)

