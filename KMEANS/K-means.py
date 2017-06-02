from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import cv2

class Segment:
    def __init__(self,segments=5):
        self.segments=segments

    def kmeans(self,image):
        image=cv2.GaussianBlur(image,(7,7),0)

        vectorized=image.reshape(-1,3)
        vectorized=np.float32(vectorized)
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(vectorized,self.segments,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))

        return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)


    def extractComponent(self,image,label_image,label):
        component=np.zeros(image.shape,np.uint8)
        component[label_image==label]=image[label_image==label]
        return component



image=cv2.imread("imgs/black_kitten.jpg")

plt.figure()
plt.imshow(image)
plt.title('original image')

segment_num = 3
seg = Segment(segment_num)

label,result=seg.kmeans(image)
plt.figure()
plt.imshow(result.astype(np.uint8))
plt.title('k-means: k=' + str(segment_num))

result=seg.extractComponent(image,label,2)
plt.figure()
plt.imshow(result.astype(np.uint8))
plt.title('extract component')



for n in [2,5,10,20,50,]:
    segment_num = n
    seg = Segment(segment_num)

    label,result=seg.kmeans(image)
    plt.figure()
    plt.imshow(result.astype(np.uint8))
    plt.title('k-means: k=' + str(segment_num))



imglist = []
for i in range(2):
    segment_num = 10
    seg = Segment(segment_num)

    label,result=seg.kmeans(image)

    imglist.append(result)

    plt.figure()
    plt.imshow(result.astype(np.uint8))
    plt.title('k-means: k=' + str(segment_num) + ' repetition: ' + str(i))

img = imglist[1] - imglist[0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img.astype(np.uint8), plt.cm.gray)
plt.title('Difference')

