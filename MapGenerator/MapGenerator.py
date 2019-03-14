
import os
import numpy as np
import pix2pix
from  data_loader import DataLoader
### picture directories as str
picspath = str('C:\\Users\\ruair\\Documents\\rolerball\\screenshotOverhead\\')


print('creating Pix2Pix model')

#create gan model 
model = pix2pix.Pix2Pix()
## that was easy, pix2pix and dataloader modified from Keras-GAN and keras-contrib respectively

model.train(epochs = 4 , batch_size = 2, datapath = picspath)


##define model optimiser 


#testimage, ignore = image_batcher(picspath,mapspath,fileList[1])
