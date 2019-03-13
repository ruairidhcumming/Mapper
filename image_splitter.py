from PIL import Image
import os
import time

picspath = str('C:\\Users\\ruair\\Documents\\rolerball\\screenshotOverhead\\')
leftpath = str('C:\\Users\\ruair\\Documents\\rolerball\\Image_X\\')
rightpath = str('C:\\Users\\ruair\\Documents\\rolerball\\Image_Y\\')

pics =  os.listdir(picspath)  
print(pics)
count = 0
timer = time.time()
for i in pics:

    print (i)
    infile = picspath + str(i)
    img = Image.open(infile)
    imgwidth, imgheight = img.size
    leftimg = img.crop((0,0,imgwidth//2,imgheight))
    rightimg = img.crop((imgwidth//2,0,imgwidth,imgheight))
    leftimg.save(leftpath+str(i))
    rightimg.save(rightpath+str(i))
    count = count + 1

timer = time.time()-timer
print (str(count)+' images split in '+str(timer))