from keras.optimizers import Adadelta

from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense,LeakyReLU,UpSampling2D
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
### picture directories as str
picspath = str('C:\\Users\\ruair\\Documents\\rolerball\\Image_X\\')
mapspath =  str('C:\\Users\\ruair\\Documents\\rolerball\\Image_Y\\')
###

batch_size = 24

###
print('creating CNN model')

model = Sequential()## lets try U- net style model
model.add(Conv2D(32, (3, 3), input_shape=( 150, 150,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))


##now conv  and upsample back to 150,150,3
model.add(UpSampling2D())
#model.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))


model.add(Conv2D(3,(3,3)))

model.add(Activation('relu'))

#model.add(Conv2DTranspose(32, kernel_size=10, strides=2, padding='same'))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.01))
model.summary()



fileList = os.listdir(picspath)     ## matching x/y in directories should have the same file name


def read_image(path ,target_width, target_height):
    img = load_img(path, target_size=( target_width, target_height))
    tmp = img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    tmp = preprocess_input(tmp)
    tmp=tmp*1./225
    return tmp

def image_batcher (Xdir,Ydir, imagelist):
    X_holder = np.zeros((len(imagelist), 150, 150, 3))## import image size to fit input graph dimensions
    Y_holder = np.zeros((len(imagelist), 118, 118, 3))## import image size to fit output graph dimensions

    for j in range(0,len(imagelist)):
        X_holder[j, :] = read_image(Xdir+imagelist[j],150, 150)
        Y_holder[j, :] = read_image(Ydir+imagelist[j], 118, 118)

    return (X_holder,Y_holder)

trainX,trainY = image_batcher('C:\\Users\\ruair\\Documents\\rolerball\\Image_X\\','C:\\Users\\ruair\\Documents\\rolerball\\Image_Y\\',fileList)

##define model optimiser 



model.compile(optimizer=Adadelta(lr = 0.2), loss = 'mse'  )
model.fit(trainX,trainY, batch_size = batch_size,epochs =1000)  
#testimage, ignore = image_batcher(picspath,mapspath,fileList[1])
testimage = trainX[1:2,:,:,:]
ans = model.predict(testimage,batch_size = 1 ,  verbose = 1)
ans = array_to_img(ans[0,:,:,:])
print(ans)

ans.show()