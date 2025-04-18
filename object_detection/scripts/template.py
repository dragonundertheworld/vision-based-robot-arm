# In[导入库]
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# In[待调节参数]
scale = 63/69.35
ori_of_world = np.array((333,340))
src_list = [(221,311), (146,477), (535,479), (474,311)] #四个顶点  左上、左下、右下、右上  
# In[获取图片]
# cap = cv2.VideoCapture(1) #捕获摄像头， 0为默认摄像头， 1为外接摄像头q
# while True:
#     ret, frame = cap.read()
#     #显示图像窗口
#     cv2.imshow("frame",frame) #显示视频图像
#     cv2.imwrite(r".\\"+ str(0) + ".jpg",frame) #保存视频帧图片
#     if cv2.waitKey(1)==ord("q"):
#         cap.release()
#         cv2.destroyAllWindows()
#         break #按下“q”按键退出
img = cv2.imread(r'.\0.JPG',1) #读取图片
# In[手动获取图中的a4纸的四个点]
def on_EVENT_LBUTTONDOWN1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print (xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img)
 
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN1)
cv2.imshow("image", img)
 
 
while(True):       
    if cv2.waitKey(1)==ord("q"):
        cv2.destroyAllWindows()
        break
# In[将图中a4纸转换视角]

pts1 = np.float32(src_list) #原始图像四个顶点
pts2 = np.float32([[158,270], [158,480], [511,480], [511,270]]) #转换目标点 图像变换尽量小一点试试
width = int(img.shape[1])
height = int(img.shape[0])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (width,height)) #透视变换
kernel = np.ones((1,1), np.uint8)
result = cv2.dilate(result, kernel)


print('Dimensions : ',result.shape)
# In[]
# 转换为HSV颜色空间
hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

# 定义红色、绿色和蓝色的颜色范围
green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])
blue_lower = np.array([100, 100, 100])
blue_upper = np.array([130, 255, 255])

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
lower_red2 = np.array([160,50,50])
upper_red2 = np.array([180,255,255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 + mask2
# red_mask = cv2.inRange(hsv, red_lower, red_upper)
green_mask = cv2.inRange(hsv, green_lower, green_upper)
blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

# 对掩模进行形态学操作，去除噪点和孔洞
kernel = np.ones((1,1),np.uint8)
red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)
green_mask = cv2.morphologyEx(green_mask,cv2.MORPH_OPEN,kernel)
blue_mask = cv2.morphologyEx(blue_mask,cv2.MORPH_OPEN,kernel)

# 找到掩模中的轮廓
red_contours,_=cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
green_contours,_=cv2.findContours(green_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
blue_contours,_=cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# In[]
# 绘制轮廓并标注颜色
for c in red_contours:
    x,y,w,h=cv2.boundingRect(c) # 获取外接矩形的坐标和宽高
    if w*h>500: # 过滤掉太小的区域
        cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),3) # 在原图上绘制红色矩形框
        cv2.putText(result,"Red",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,1,(0 ,0 ,255),3) # 在原图上标注"Red"

for c in green_contours:
    x,y,w,h=cv2.boundingRect(c) 
    if w*h>500: 
        cv2.rectangle(result,(x,y),(x+w,y+h),(0 ,255 ,0),3) 
        cv2.putText(result,"Green",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,1,(0 ,255 ,0),3)

for c in blue_contours:
    x,y,w,h=cv2.boundingRect(c) 
    if w*h>500: 
        cv2.rectangle(result,(x,y),(x+w,y+h),(255 ,0 ,0),3) 
        cv2.putText(result,"Blue",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,1,(255 ,0 ,0),3)

# 显示结果图片
cv2.drawContours(result, red_contours,-1,(0,0,255),3)
cv2.drawContours(result, green_contours,-1,(0,0,255),3)

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print (xy)
        cv2.circle(result, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(result, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("result", result)
 
 
cv2.namedWindow("result")
cv2.setMouseCallback("result", on_EVENT_LBUTTONDOWN)
 
 
while(True):       
    if cv2.waitKey(1)==ord("q"):
        cv2.destroyAllWindows()
        break
# In[]
def get_contour_max_list(contours):# contours不只包括大轮廓，还有个别小轮廓
    contour_max_list = []
    for dart in contours:
        x,y,w,h=cv2.boundingRect(dart)
        if w*h>3000:
            dart = dart[:,0]
            contour_max = tuple(dart[np.argmax(dart[:,1])])
            x = contour_max[0]
            contour_max_list.append(contour_max)
    return contour_max_list,x

red_points,x_red = get_contour_max_list(red_contours)
green_points,x_green = get_contour_max_list(green_contours)
print('max point of red:',red_points)
print('max point of green:',green_points)
# In[]
# test_point = [(322,314),(313,321),(305,329),(298,335),(289,345),(281,352),(274,360)]
theta = -42/180*(np.pi)# -60
sin = np.sin(theta)
cos = np.cos(theta)
rot_mat = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])

Paorg_b = np.hstack((ori_of_world , np.array([0])))
Pborg_a = -np.dot(rot_mat,Paorg_b)

tran_mat_temp = np.hstack((rot_mat,Pborg_a.reshape(3,1)))
tran_mat = np.concatenate((tran_mat_temp,np.array([0,0,0,1]).reshape(1,4)),axis=0)
print('变换矩阵：\n',tran_mat)

red_point_list = []
green_point_list = []

def python2world(points,color_point_list):
    for ele in points:
        ele = ele + (0,1)
        final_point = np.dot(tran_mat,ele)[:2]*scale # 左乘变换矩阵
        print('final_point:',final_point)
        x =  36.2226 + 0.9452*final_point[0] + 0.0786*final_point[1]  + 204 #+ (-0.0004)*final_point[0]*final_point[0]
        y = 12.5045 + 0.0226*final_point[0] + 0.9213*final_point[1] + 0.0003*final_point[1]*final_point[1]+ 204 # + -0.0001*final_point[0]*final_point[0]
        if x>147 and y>274 :
            x = x + 5
            y = y + 5
        # x = final_point[0]
        # y = final_point[1]
        final_point= (x,y)# +186
        color_point_list.append(final_point)
    return color_point_list
        
green_point_list = python2world(green_points,green_point_list)
red_point_list = python2world(red_points,red_point_list)

print('final red point list:',red_point_list)
print('final green point list:',green_point_list) # 画的坐标系下表示

# In[准备传输数据]
all_point_list = []
mid_red_point = red_point_list.copy()
mid_green_point = green_point_list.copy()

mid_red_point.remove(min(red_point_list))
mid_red_point.remove(max(red_point_list))

mid_green_point.remove(min(green_point_list))
mid_green_point.remove(max(green_point_list))


#按照x坐标改变顺序
red_point_list = [max(red_point_list),mid_red_point[0],min(red_point_list)]
green_point_list = [max(green_point_list),mid_green_point[0],min(green_point_list)]
for i in range(len(red_point_list)):
    all_point_list.append(red_point_list[i])
    all_point_list.append(green_point_list[i])
control_data = []

i = 0
for point in all_point_list:
    if i%2 == 0:
        control_data.append((str(int(point[0]))+'x',str(int(point[1]))+'y','10z'))
    else:
        control_data.append((str(int(point[0]))+'x',str(int(point[1]))+'y','20z'))
    i = i+1
print(control_data)
# # In[串口通信]
# import serial
# import time
# # 连接串口
# serial = serial.Serial('COM3', 115200, timeout = 0.01)
# i=0
# if serial.isOpen():
#     print('串口已打开')
#     time.sleep(3)
#     for data in control_data:
#         # if i <1:
#         serial.write(data[0].encode()) # 串口写x
#         serial.write(data[0].encode())
#         print('data[0]:',data[0])
#         serial.write(data[1].encode()) # 串口写y
#         serial.write(data[1].encode()) # 串口写y
#         print('data[1]:',data[1])
#         serial.write(data[2].encode()) # 串口写z
#         serial.write(data[2].encode()) # 串口写z
#         print('data[2]:',data[2])
#         time.sleep(16)
# else:
    # print('串口未打开')