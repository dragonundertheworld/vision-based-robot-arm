# color_dectection
import cv2
import numpy as np
import matplotlib.pyplot as plt

# In[]
img = cv2.imread(r'.\3.JPG',1)
# print('源图：------------------------------------')
# plt.imshow(img)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #HSV空间，图片为RGB格式
# In[]
def myplot(images,titles): # For plotting multiple images at once
    fig, axs=plt.subplots(1,len(images),sharey=True)# 创造空图
    fig.set_figwidth(15)
    for img,ax,title in zip(images,axs,titles):
        if img.shape[-1]==3:# 如果是3个维度则转换成RGB
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV reads images as BGR, so converting back them to RGB
            print('RGB转换完成')
        else:
            print(img.shape)
            img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)# 如果是一个维度(即灰度图)则转换成BGR
            print('灰度图到BGR转换完成')
        ax.imshow(img)
        ax.set_title(title)
# In[blue]
lower_blue= np.array([110,100,100]) #设定蓝色的阈值
upper_blue = np.array([130,255,255])
blue_mask = cv2.inRange(hsv,lower_blue,upper_blue) #设定取值范围设定取值范围
blue = cv2.bitwise_and(img,img,mask=blue_mask)
myplot([img,blue], ['Original Image', 'blue'])
dart_blue = blue.copy() #复制图像复制图像
kernel = np.ones((3,3),np.uint8)
erode1=cv2.erode(blue,kernel,dart_blue,iterations=20) #腐蚀运算腐蚀运算
print('shape of erode1',erode1.shape)
plt.imshow(erode1)
myplot([blue,erode1], ['blue','dart'])
# In[green]

lower_green = np.array([35,50,50]) #设定绿色的阈值
upper_green = np.array([77,255,255])
green_mask = cv2.inRange(hsv,lower_green,upper_green) #设定取值范围设定取值范围
green = cv2.bitwise_and(img,img,mask=green_mask)
myplot([img,green], ['Original Image', 'green'])
dart_green = green.copy() #复制图像复制图像
kernel = np.ones((3,3),np.uint8)
erode1=cv2.erode(green,kernel,dart_green,iterations=20) #腐蚀运算腐蚀运算
plt.imshow(erode1)
myplot([green,erode1], ['green','dart'])

# In[yellow]
lower_yellow= np.array([26,50,50]) #设定黄色的阈值
upper_yellow = np.array([34,255,255])
yellow_mask = cv2.inRange(hsv,lower_yellow,upper_yellow) #设定取值范围设定取值范围
yellow = cv2.bitwise_and(img,img,mask=yellow_mask)
myplot([img,yellow], ['Original Image', 'yellow'])
dart_yellow = yellow.copy() #复制图像复制图像
kernel = np.ones((3,3),np.uint8)
erode1=cv2.erode(yellow,kernel,dart_yellow,iterations=20) #腐蚀运算腐蚀运算
plt.imshow(erode1)
myplot([yellow,erode1], ['yellow','dart'])

# In[red]
lower_redandgreen = np.array([0,50,50]) #设定红色+绿色的阈值
upper_redandgreen = np.array([77,255,255])
redandgreen_mask = cv2.inRange(hsv,lower_redandgreen,upper_redandgreen) #设定取值范围设定取值范围
redandgreen = cv2.bitwise_and(img,img,mask=redandgreen_mask)

myplot([img,redandgreen], ['Original Image', 'red and green'])
dart = redandgreen.copy() #复制图像复制图像
kernel = np.ones((3,3),np.uint8)
erode1=cv2.erode(redandgreen,kernel,dart,iterations=20) #腐蚀运算腐蚀运算
plt.imshow(erode1)
myplot([redandgreen,erode1], ['red and green','dart'])

# In[]

img_gray=cv2.cvtColor(erode1,cv2.COLOR_BGR2GRAY) #灰度，图片为灰度，图片为RGB格式格式
_,img_binary=cv2.threshold(img_gray,50,200,cv2.THRESH_BINARY) #二值化图像二值化图像。可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像
contours,hierarchy=cv2.findContours(img_binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #边缘检测
img_contour = cv2.drawContours(img, contours,-1,(0,0,255),3) # Draws the contours on the original image just like draw function
# cv2.imshow('img_contour',img_contour)
myplot([img_binary, img], ['Binary Image', 'Contours in the Image'])

# In[]
print('contours shape',len(contours))
dart0= contours[0]
dart1= contours[1]
dart2= contours[2]
print(dart0.shape)

# In[]
dart0 = dart0[:,0]
dart1 = dart1[:,0]
dart2 = dart2[:,0]
print(dart0.shape)

print(dart0.mean(axis = 0)) #取平均值，获取坐标取平均值，获取坐标
print(dart1.mean(axis = 0))
print(dart2.mean(axis = 0))