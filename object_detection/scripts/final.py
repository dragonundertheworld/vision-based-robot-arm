# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:43:12 2023

@author: zhihan
"""
# In[导入库]
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[拍摄图像]
if(1==0):
    cap = cv2.VideoCapture(1) #捕获摄像头， 0为默认摄像头， 1为外接摄像头
    print(type(cap))
    while True:
        ret, frame = cap.read()
        #显示图像窗口
        cv2.imshow("frame",frame) #显示视频图像
        cv2.imwrite(r".\\"+ str(1) + ".jpg",frame) #保存视频帧图片
        if cv2.waitKey(1)==ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break #按下“q”按键退出
# In[图像读取与预处理]
img = cv2.imread(r'.\1.JPG',1) #读取图片

a = np.arange(5, 16, 5) / 10
b = np.arange(-30, 31, 30)
bri_mean = np.mean(img)
a_len = len(a)
b_len = len(b)
print(a_len, b_len)
plt.figure()

for i in range(a_len):
    for j in range(b_len):
        aa = a[i]
        bb = b[j]
        img_a = aa * (img-bri_mean) + bb + bri_mean
        print(i, j, aa, bb)
        img_a = np.clip(img_a,0,255).astype(np.uint8)
        plt.subplot(a_len+1, b_len, (j + b_len * i + 1))
        plt.imshow(img_a, cmap='gray')
plt.subplot(a_len + 1, b_len, a_len*b_len+1)
plt.imshow(img.astype(np.uint8), cmap='gray')
plt.show()
# In[调整图像大小]
scale_percent = 20 # 图片缩小比例
width = int(img_a.shape[1] * scale_percent / 100)
height = int(img_a.shape[0] * 30 / 100)
dim = (width, height)
img_a = cv2.resize(img_a, dim, interpolation = cv2.INTER_AREA)
#图片尺寸 过大会影响处理效率，所以进行缩放
print('Resized Dimensions : ',img_a.shape)
# In[手动获取图中的a4纸的四个点]
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print (xy)
        cv2.circle(img_a, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img_a, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img_a)
 
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img_a)
 
 
while(True):       
    if cv2.waitKey(1)==ord("q"):
        cv2.destroyAllWindows()
        break

# In[将图中a4纸转换视角]
src_list = [(50, 47), (17, 134), (122, 139), (93, 46)] #四个顶点  左上、左下、右下、右上
# for i, pt in enumerate(src_list):
#     cv2.circle(img_a, pt, 5, (0, 0, 255), -1)
#     cv2.putText(img_a,str(i+1),(pt[0]+5,pt[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
pts1 = np.float32(src_list) #原始图像四个顶点
pts2 = np.float32([[20, 11], [20, 131], [105, 131], [105, 11]]) #转换目标点

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img_a, matrix, ( width,height)) #透视变换
# In[接转换视角]
print('Dimensions : ',result.shape)
cv2.imshow("Image", img_a)
cv2.imshow("Perspective transformation", result)
while(True):       
    if cv2.waitKey(1)==ord("q"):
        cv2.destroyAllWindows()
        break
# In[color_dectection]
# In[BGR2HSV]
hsv=cv2.cvtColor(result,cv2.COLOR_BGR2HSV) #HSV空间，图片为RGB格式
# In[]
def myplot(images,titles): # For plotting multiple images at once
    fig, axs=plt.subplots(1,len(images),sharey=True)# 创造空图
    fig.set_figwidth(15)
    for img_a,ax,title in zip(images,axs,titles):
        if img_a.shape[-1]==3:# 如果是3个维度则转换成RGB
            img_a=cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB) # OpenCV reads images as BGR, so converting back them to RGB
            print('RGB转换完成')
        else:
            print(img_a.shape)
            img_a=cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)# 如果是一个维度(即灰度图)则转换成BGR
            print('灰度图到BGR转换完成')
        ax.imshow(img_a)
        ax.set_title(title)
# # In[blue]
# lower_blue= np.array([110,100,100]) #设定蓝色的阈值
# upper_blue = np.array([130,255,255])
# blue_mask = cv2.inRange(hsv,lower_blue,upper_blue) #设定取值范围设定取值范围
# blue = cv2.bitwise_and(img_a,img_a,mask=blue_mask)
# myplot([img_a,blue], ['Original Image', 'blue'])
# dart_blue = blue.copy() #复制图像复制图像
# kernel = np.ones((3,3),np.uint8)
# erode1=cv2.erode(blue,kernel,dart_blue,iterations=20) #腐蚀运算腐蚀运算
# print('shape of erode1',erode1.shape)
# plt.imshow(erode1)
# myplot([blue,erode1], ['blue','dart'])
# In[green]

lower_green = np.array([35,50,50]) #设定绿色的阈值
upper_green = np.array([77,255,255])
green_mask = cv2.inRange(hsv,lower_green,upper_green) #设定取值范围设定取值范围
green = cv2.bitwise_and(result,result,mask=green_mask)
myplot([result,green], ['Original Image', 'green'])
dart_green = green.copy() #复制图像复制图像
kernel = np.ones((1,1),np.uint8)
erode1=cv2.erode(green,kernel,dart_green,iterations=20) #腐蚀运算腐蚀运算
plt.imshow(erode1)
myplot([green,erode1], ['green','dart'])

# In[red]

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

red_mask = cv2.inRange(hsv,lower_red,upper_red) #设定取值范围
red = cv2.bitwise_and(result,result,mask=red_mask)

myplot([result,red], ['Original Image', 'red'])
dart = red.copy() #复制图像复制图像
kernel = np.ones((1,1),np.uint8)
erode2=cv2.erode(red,kernel,dart,iterations=20) #腐蚀运算
myplot([red,erode1], ['red','dart'])

# In[red&green]
lower_redandgreen = np.array([0,50,50]) #设定红色+绿色的阈值
upper_redandgreen = np.array([77,255,255])
redandgreen_mask = cv2.inRange(hsv,lower_redandgreen,upper_redandgreen) #设定取值范围设定取值范围
redandgreen = cv2.bitwise_and(result,result,mask=redandgreen_mask)

myplot([result,redandgreen], ['Original Image', 'red and green'])
dart = redandgreen.copy() #复制图像复制图像
kernel = np.ones((1,1),np.uint8)
redandgreen_mask = cv2.morphologyEx(redandgreen_mask,cv2.MORPH_OPEN,kernel)
erode3=cv2.erode(redandgreen,kernel,dart,iterations=20) #腐蚀运算腐蚀运算
myplot([redandgreen,erode3], ['red and green','dart'])
# In[]
# img_gray_erode1=cv2.cvtColor(erode1,cv2.COLOR_BGR2GRAY) #灰度，图片为灰度，图片为RGB格式格式
# img_gray_erode2=cv2.cvtColor(erode2,cv2.COLOR_BGR2GRAY)
img_gray_erode3=cv2.cvtColor(erode3,cv2.COLOR_BGR2GRAY)

# _,img_binary_erode1=cv2.threshold(img_gray_erode1,50,200,cv2.THRESH_BINARY) #二值化图像二值化图像。可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像
# _,img_binary_erode2=cv2.threshold(img_gray_erode2,50,200,cv2.THRESH_BINARY) #二值化图像二值化图像。可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像
_,img_binary_erode3=cv2.threshold(img_gray_erode3,50,200,cv2.THRESH_BINARY) #二值化图像二值化图像。可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像

# contours1,hierarchy1=cv2.findContours(img_binary_erode1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #边缘检测
# contours2,hierarchy2=cv2.findContours(img_binary_erode2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #边缘检测
contours3,hierarchy2=cv2.findContours(img_binary_erode3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #边缘检测



# img_contour1 = cv2.drawContours(result, contours1,-1,(0,0,255),3) # Draws the contours on the original image just like draw function
# img_contour2 = cv2.drawContours(result, contours2,-1,(0,0,255),3) # Draws the contours on the original image just like draw function
img_contour3 = cv2.drawContours(result, contours3,-1,(0,0,255),3) # Draws the contours on the original image just like draw function


# cv2.imshow('img_contour',img_contour)
def get_contour_max_list(contours):
    contour_max_list = []
    for dart in contours:
        dart = dart[:,0]
        print(dart)
        contour_max = tuple(dart[np.argmax(dart[:,1])])
        contour_max_list.append(contour_max)
    return contour_max_list

# contour1_max_list = get_contour_max_list(contours1)
# contour2_max_list = get_contour_max_list(contours2)
contour3_max_list = get_contour_max_list(contours3)

# print("contour1 max list:",contour1_max_list)
# print("contour2 max list:",contour2_max_list)
print("contour3 max list:",contour3_max_list)

# In[]
lst = []
def find_max_index(contour_max_list):
    for point in contour_max_list:
        lst.append(point[1])
        
    print('list:',lst)
    for i,item in enumerate(lst):
        if item == max(lst):
            break
    return i

# i1 = find_max_index(contour1_max_list)
# i2 = find_max_index(contour2_max_list)
i3 = find_max_index(contour3_max_list)

# print('max point1:',contour1_max_list[i1])
# print('max point2:',contour2_max_list[i2])
print('max point2:',contour3_max_list[i3])

myplot([img_binary_erode3, result], ['Binary Image', 'Contours in the Image'])
# In[与现实坐标转换]
scale_world = 210/(110-25)#a4纸的宽(mm)/图像中a4的宽的像素
world_loc = (contour3_max_list[i3][0]*scale_world,contour3_max_list[i3][1]*scale_world)
# # In[]
# # 定义红色、绿色和蓝色的颜色范围
# red_lower = np.array([0, 100, 100])
# red_upper = np.array([10, 255, 255])
# green_lower = np.array([40, 50, 50])
# green_upper = np.array([80, 255, 255])
# blue_lower = np.array([100, 100, 100])
# blue_upper = np.array([130, 255, 255])

# # 根据颜色范围创建掩模
# red_mask = cv2.inRange(hsv, red_lower, red_upper)
# green_mask = cv2.inRange(hsv, green_lower, green_upper)
# blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

# # 对掩模进行形态学操作，去除噪点和孔洞
# kernel = np.ones((3,3),np.uint8)
# red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)
# green_mask = cv2.morphologyEx(green_mask,cv2.MORPH_OPEN,kernel)
# blue_mask = cv2.morphologyEx(blue_mask,cv2.MORPH_OPEN,kernel)

# # 找到掩模中的轮廓
# red_contours,_=cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# green_contours,_=cv2.findContours(green_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# img_contour_new = cv2.drawContours(result, red_contours,-1,(0,0,255),3) # Draws the contours on the original image just like draw function
# cv2.imshow('new',img_contour_new)


