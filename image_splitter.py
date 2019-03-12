from PIL import Image
import os

picspath = str('C:\\Users\\ruair\\Documents\\rolerball\\screenshotOverhead\\')
leftpath = str('C:\\Users\\ruair\\Documents\\rolerball\\Image_X\\')
rightpath = str('C:\\Users\\ruair\\Documents\\rolerball\\Image_Y\\')

pics =  os.listdir(picspath)  
print(pics)
count = 0

for i in pics[0:10]:

    print (i)
    infile = picspath + str(i)
    img = Image.open(infile)
    imgwidth, imgheight = img.size
    leftimg = img.crop((0,0,imgwidth//2,imgheight))
    rightimg = img.crop((imgwidth//2,0,imgwidth,imgheight))
    leftimg.show()
    rightimg.show()
    count = count + 1

    
print (str(count)+' images used')