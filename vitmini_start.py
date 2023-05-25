# encoding=utf-8
# encoding=utf-8
import common.data_io.SsrLoader as ssrloader
from common.Control import Control
from common.Streaming import Data
from common.Stream import Streaming
from common.Streaming.BlobServerConfiguration import BlobClientConfig
import cv2
import numpy as np

# 设置putText函数字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    squares = []
    center=[]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    bin = cv2.Canny(img, 30, 100, apertureSize=3)
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        # print('1111111111')
        # print(cv2.contourArea(cnt))
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.1:
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                # cv2.putText(img, ("#%d" % index), (cx, cy), font, 0.7, (255, 0, 255), 2)
                squares.append(cnt)
                center.append([cx,cy])
    return center,squares, img

class VITMINI:

    def __init__(self, ip, control_port=2122, streaming_port=2114, protocol="Cola2"):
        self.ip = ip
        self.control_port = control_port
        self.streaming_port = streaming_port
        self.protocol = protocol

    def init_camera(self):
        # create and open a control connection to the device
        self.deviceControl = Control(self.ip, self.protocol, self.control_port)
        self.deviceControl.open()
        self.deviceControl.singleStep()
        self.streaming_device = Streaming(self.ip, self.streaming_port)


    def get_sn(self):
        name, version = self.deviceControl.getIdent()
        sn=self.deviceControl.getSerialNumber()
        # sn=b'22190099'
        return (name.decode('utf-8'))+(version.decode('utf-8'))+(sn.decode('utf-8'))


    def get_continue_data(self):
        try:
            while True:
                self.streaming_device.getFrame()
                wholeFrame = self.streaming_device.frame
                self.myData.read(wholeFrame)
                if self.myData.hasDepthMap:
                    intensityData = self.myData.depthmap.intensity
                    distanceData = self.myData.depthmap.distance
                    numCols = self.myData.cameraParams.width
                    numRows = self.myData.cameraParams.height
        except KeyboardInterrupt:
            print("")
            print("Terminating")
        except Exception as e:
            print(f"Exception -{e.args[0]}- occurred, check your device configuration")

    def get_single_data(self):
        try:
            self.streaming_device.openStream()

            # request the whole frame data
            self.streaming_device.getFrame()

            # access the new frame via the corresponding class attribute
            wholeFrame = self.streaming_device.frame

            # create new Data object to hold/parse/handle frame data
            myData = Data.Data()
            # parse frame data to Data object for further data processing
            myData.read(wholeFrame)
            numCols = myData.cameraParams.width
            numRows = myData.cameraParams.height
            intensityData = myData.depthmap.intensity
            distanceData = myData.depthmap.distance
            distanceData = np.reshape(distanceData, (numRows, numCols))

            intensityDataArray = np.uint16(np.reshape(intensityData, (numRows, numCols)))
            intensityDataArray[intensityDataArray < 20000] = 255
            intensityDataArray[intensityDataArray == 20000] = 0
            #  0是黑色 255 白色
            # uint16_img = np.clip(intensityDataArray, 0, 2000)
            uint8_img = np.uint8(intensityDataArray)
            center,squares, img = find_squares(uint8_img)
            im_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(im_color, squares, -1, (255, 0, 0), 2)
            # x,y
            x=center[0][0]
            y=center[0][1]

            sq=np.array(squares).reshape(4,2)
            x_max=np.max(sq[:,0])
            x_min=np.min(sq[:,0])
            y_max=np.max(sq[:,1])
            y_min=np.min(sq[:,1])
            #  求范围均值
            darray=distanceData[(y_min):(y_max),(x_min):(x_max)]
            darray=darray.flatten()
            mask = darray != 0  # [False False False  True  True  True]
            new_data = darray[mask]  # [135  30 125]
            d=np.mean(new_data)

            #  求左上角和右下角均值
            # d1=distanceData[y_min-2][x_min-2]
            # d2=distanceData[y_max+2][x_max+2]
            # print(d1,d2)
            # if d1!=0 and d2!=0:
            #     d=np.mean([d1,d2])
            # elif d1!=0:
            #     d=d1
            # else:
            #     d=d2


            data=self.new_cam2word(np.array([[x,y,d]]),myData)
            print('x=%d,y=%d,z=%d,d=%d'%(data[0][0],data[0][1],data[0][2],data[0][3]))
            cv2.imshow('squares', im_color)
            cv2.waitKey(1)
            # find_squares(uint8_img)
            # im_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('i.jpg',im_color)

            # save img
            # if result[0][3]:
            #     import datetime
            #     filename = datetime.datetime.now().strftime('%d_%H_%M_%S') + '.jpg'
            #     cv2.imwrite(filename, im_color)

            if myData.hasDepthMap:
                self.streaming_device.closeStream()
                # self.deviceControl.startStream()
                return myData
        except Exception as e:
            self.streaming_device.closeStream()
            # self.deviceControl.startStream()
            # self.deviceControl.initStream()
            print(e)

    def close(self):
        self.streaming_device.closeStream()
        self.deviceControl.startStream()
    #  todo  传进来矩阵[[x,y,z],[x,y,z]]
    def cam2word(self,box_list,myData):
        stereo=myData.xmlParser.stereo
        box_list=np.array(box_list)
        cx = myData.cameraParams.cx
        fx = myData.cameraParams.fx
        cy = myData.cameraParams.cy
        fy = myData.cameraParams.fy
        m_c2w = myData.cameraParams.cam2worldMatrix
        xp = (cx - box_list[:,0]) / fx
        yp = (cy - box_list[:,1]) / fy
        xc = xp * box_list[:,2]
        yc = yp * box_list[:,2]
        box_list[:,0] = m_c2w[3] +box_list[:,2] *m_c2w[2] +yc *m_c2w[1] +xc *m_c2w[0]
        box_list[:,1] = m_c2w[7] +box_list[:,2] *m_c2w[6] +yc *m_c2w[5] +xc *m_c2w[4]
        box_list[:,2] = m_c2w[11] +box_list[:,2] *m_c2w[10] +yc *m_c2w[9] +xc *m_c2w[8]

        return box_list

    def new_cam2word(self,box_list,myData):
        cx = myData.cameraParams.cx
        fx = myData.cameraParams.fx
        cy = myData.cameraParams.cy
        fy = myData.cameraParams.fy
        m_c2w = myData.cameraParams.cam2worldMatrix
        xp = (cx - box_list[:,0]) / fx
        yp = (cy - box_list[:,1]) / fy

        r2 = (xp * xp + yp * yp)
        r4 = r2 * r2

        k = 1 + myData.cameraParams.k1 * r2 + myData.cameraParams.k2 * r4

        xd = xp * k
        yd = yp * k

        d = box_list[:,2].copy()
        s0 = np.sqrt(xd * xd + yd * yd + 1)

        xc = xd * d / s0
        yc = yd * d / s0
        zc = d / s0 - myData.cameraParams.f2rc

        box_list[:,0] = m_c2w[3] +zc *m_c2w[2] +yc *m_c2w[1] +xc *m_c2w[0]
        box_list[:,1] = m_c2w[7] +zc *m_c2w[6] +yc *m_c2w[5] +xc *m_c2w[4]
        box_list[:,2] = m_c2w[11] +zc*m_c2w[10] +yc *m_c2w[9] +xc *m_c2w[8]

        return np.column_stack((box_list, d))

vit=VITMINI('192.168.0.50')
vit.init_camera()
while 1:
    vit.get_single_data()