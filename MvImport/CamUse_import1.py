import os
import random
import sys
import threading
import time
import tkinter as tk
import numpy as np
import cv2
from os import getcwd
import msvcrt
from ctypes import *
from setuptools import setup
from setuptools.extension import Extension
from pathlib import Path
import shutil
import threading
from queue import Queue

# sys.path.append("")
from MvCameraControl_class import *

sys.path.append('D:\\Project\\project 1')
import main


def enum_devices(device=0, device_way=False):
    """
    查看电脑上的设备，返回deviceList
    device = 0  枚举网口、USB口、未知设备、cameralink 设备
    device = 1 枚举GenTL设备
    """
    if device_way == False:
        if device == 0:
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_UNKNOW_DEVICE | MV_1394_DEVICE | MV_CAMERALINK_DEVICE
            deviceList = MV_CC_DEVICE_INFO_LIST()
            # 枚举设备
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                print("enum devices fail! ret[0x%x]" % ret)
                sys.exit()
            if deviceList.nDeviceNum == 0:
                print("find no device!")
                sys.exit()
            print("Find %d devices!" % deviceList.nDeviceNum)
            return deviceList
        else:
            pass
    elif device_way == True:
        pass


def identify_different_devices(deviceList):
    """
    根据diviceList,查看每个设备的详细信息
    """
    # 判断不同类型设备，并输出相关信息
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        # 判断是否为网口相机
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\n网口设备序号: [%d]" % i)
            # 获取设备名
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("当前设备型号名: %s" % strModeName)
            # 获取当前设备 IP 地址
            nip1_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip1_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip1_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip1_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("当前 ip 地址: %d.%d.%d.%d" % (nip1_1, nip1_2, nip1_3, nip1_4))
            # 获取当前子网掩码
            nip2_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0xff000000) >> 24)
            nip2_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x00ff0000) >> 16)
            nip2_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x0000ff00) >> 8)
            nip2_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x000000ff)
            print("当前子网掩码 : %d.%d.%d.%d" % (nip2_1, nip2_2, nip2_3, nip2_4))
            # 获取当前网关
            nip3_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0xff000000) >> 24)
            nip3_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x00ff0000) >> 16)
            nip3_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x0000ff00) >> 8)
            nip3_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x000000ff)
            print("当前网关 : %d.%d.%d.%d" % (nip3_1, nip3_2, nip3_3, nip3_4))
            # 获取网口 IP 地址
            nip4_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0xff000000) >> 24)
            nip4_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x00ff0000) >> 16)
            nip4_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x0000ff00) >> 8)
            nip4_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x000000ff)
            print("当前连接的网口 IP 地址 : %d.%d.%d.%d" % (nip4_1, nip4_2, nip4_3, nip4_4))
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)
            # 获取制造商的具体信息
            stManufacturerSpecificInfo = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerSpecificInfo:
                stManufacturerSpecificInfo = stManufacturerSpecificInfo + chr(per)
            print("设备制造商的具体信息 : %s" % stManufacturerSpecificInfo)
            # 获取设备序列号
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            print("设备序列号 : %s" % stSerialNumber)
            # 获取用户自定义名称
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
            print("用户自定义名称 : %s" % stUserDefinedName)

        # 判断是否为 USB 接口相机
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nU3V 设备序号e: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("当前设备型号名 : %s" % strModeName)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("当前设备序列号 : %s" % strSerialNumber)
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)
            # 获取设备序列号
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            print("设备序列号 : %s" % stSerialNumber)
            # 获取用户自定义名称
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
            print("用户自定义名称 : %s" % stUserDefinedName)
            # 获取设备 GUID
            stDeviceGUID = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceGUID:
                stDeviceGUID = stDeviceGUID + chr(per)
            print("设备GUID号 : %s" % stDeviceGUID)
            # 获取设备的家族名称
            stFamilyName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chFamilyName:
                stFamilyName = stFamilyName + chr(per)
            print("设备的家族名称 : %s" % stFamilyName)

        # 判断是否为 1394-a/b 设备
        elif mvcc_dev_info.nTLayerType == MV_1394_DEVICE:
            print("\n1394-a/b device: [%d]" % i)

        # 判断是否为 cameralink 设备
        elif mvcc_dev_info.nTLayerType == MV_CAMERALINK_DEVICE:
            print("\ncameralink device: [%d]" % i)
            # 获取当前设备名
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("当前设备型号名 : %s" % strModeName)
            # 获取当前设备序列号
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("当前设备序列号 : %s" % strSerialNumber)
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)


def input_num_camera(deviceList):
    """
    选择设备
    """
    nConnectionNum = input("please input the number of the device to connect:")
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()
    return nConnectionNum


def creat_camera(deviceList, nConnectionNum):
    """
    根据选择的设备号创建相机
    """
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()
    return cam, stDeviceList


def open_device(cam, stDeviceList):
    """
    打开设备，并获取数据包大小
    """
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue
    return nPayloadSize


# 获取各种类型节点参数
def get_Value(cam, param_type="int_value", node_name="PayloadSize"):
    """
    获取各种值的方法
    """
    if param_type == "int_value":
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = cam.MV_CC_GetIntValueEx(node_name, stParam)
        if ret != 0:
            print("获取 int 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        int_value = stParam.nCurValue
        return int_value

    elif param_type == "float_value":
        stFloatValue = MVCC_FLOATVALUE()
        memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
        ret = cam.MV_CC_GetFloatValue(node_name, stFloatValue)
        if ret != 0:
            print("获取 float 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        float_value = stFloatValue.fCurValue
        return float_value

    elif param_type == "enum_value":
        stEnumValue = MVCC_ENUMVALUE()
        memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
        ret = cam.MV_CC_GetEnumValue(node_name, stEnumValue)
        if ret != 0:
            print("获取 enum 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        enum_value = stEnumValue.nCurValue
        return enum_value

    elif param_type == "bool_value":
        stBool = c_bool(False)
        ret = cam.MV_CC_GetBoolValue(node_name, stBool)
        if ret != 0:
            print("获取 bool 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        return stBool.value

    elif param_type == "string_value":
        stStringValue = MVCC_STRINGVALUE()
        memset(byref(stStringValue), 0, sizeof(MVCC_STRINGVALUE))
        ret = cam.MV_CC_GetStringValue(node_name, stStringValue)
        if ret != 0:
            print("获取 string 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        string_value = stStringValue.chCurValue
        return string_value


# 设置各种类型节点参数
def set_Value(cam, param_type="int_value", node_name="PayloadSize", node_value=None):
    """
    :param cam:               相机实例
    :param param_type:        需要设置的节点值得类型
        int:
        float:
        enum:     参考于客户端中该选项的 Enum Entry Value 值即可
        bool:     对应 0 为关，1 为开
        string:   输入值为数字或者英文字符，不能为汉字
    :param node_name:         需要设置的节点名
    :param node_value:        设置给节点的值
    :return:
    """
    if param_type == "int_value":
        stParam = int(node_value)
        ret = cam.MV_CC_SetIntValueEx(node_name, stParam)
        if ret != 0:
            print("设置 int 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        print("设置 int 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))

    elif param_type == "float_value":
        stFloatValue = float(node_value)
        ret = cam.MV_CC_SetFloatValue(node_name, stFloatValue)
        if ret != 0:
            print("设置 float 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        print("设置 float 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))

    elif param_type == "enum_value":
        stEnumValue = node_value
        ret = cam.MV_CC_SetEnumValue(node_name, stEnumValue)
        if ret != 0:
            print("设置 enum 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        print("设置 enum 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))

    elif param_type == "bool_value":
        ret = cam.MV_CC_SetBoolValue(node_name, node_value)
        if ret != 0:
            print("设置 bool 型数据节点 %s 失败 ！ 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        print("设置 bool 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))

    elif param_type == "string_value":
        stStringValue = str(node_value)
        ret = cam.MV_CC_SetStringValue(node_name, stStringValue)
        if ret != 0:
            print("设置 string 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
            sys.exit()
        print("设置 string 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))


def start_grabing(cam):
    """
    开始推流
    """
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()


def image_control(data, stFrameInfo):
    """
    处理照片
    """
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    elif stFrameInfo.enPixelType == 17301512:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2BGR)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif stFrameInfo.enPixelType == 34603039:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
    return image


def access_get_image(cam, nPayloadSize):
    """
    主动图像采集
    """
    global data_buf

    stParam = MVCC_INTVALUE_EX()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    data_buf = (c_ubyte * nPayloadSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d] " % (
                stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))

            image = np.asarray(data_buf)
            img = image_control(data=image, stFrameInfo=stFrameInfo)
            return img

        else:
            print("no data[0x%x]" % ret)


def stop_grabing(cam):
    """
    结束推流
    """
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()


def close_device(cam):
    """
    关闭设备
    """
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()


def destroy_handle(cam):
    """
    摧毁句柄
    """
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()


def startCam():
    """
    打开设备的最终接口
    """
    deviceList = enum_devices(device=0, device_way=False)
    identify_different_devices(deviceList)
    nConnectionNum = 0
    cam, stDeviceList = creat_camera(deviceList, nConnectionNum)
    nPayloadSize = open_device(cam, stDeviceList)
    start_grabing(cam)
    return cam, nPayloadSize


def stopCam(cam):
    """
    关闭设备的最终接口
    """
    stop_grabing(cam)
    close_device(cam)
    destroy_handle(cam)


class machine(threading.Thread):
    def run(self):
        global x
        while True:
            x = input('>>')
            if x == '2':
                with lock_con1:
                    lock_con1.notify()
                print('machine exit')
                break
            elif x == '1':
                print('信号传输')
                with lock_con1:
                    lock_con1.notify()
                print('信号传递结束')
                x = 0
                event1.clear()
                event1.wait()


class take_photo(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        global x
        while True:
            with lock_con1:
                lock_con1.wait()
            if x == '2':
                self.queue.put('end')
                print('take_photo exit')
                break
            else:
                print('开始拍摄')
                img1 = access_get_image(cam, nPayloadSize)
                self.queue.put(img1)
                event1.set()
                print('拍摄结束')


class process(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            img = self.queue.get()
            if img == 'end':
                print('process exit')
                break
            else:
                print('开始处理')
                cv2.imshow('img', img)
                cv2.waitKey(0)
                # time.sleep(1)
                self.queue.task_done()
                print('处理结束')


if __name__ == "__main__":
    temp = cv2.imread('D:\Project\project 1\img\ws03\Image_20220517145136166.bmp')
    lock_con1 = threading.Condition()
    event1 = threading.Event()
    Lock = threading.Lock()
    threads = []
    x = 0
    queue = Queue()
    cam, nPayloadSize = startCam()
    threads.append(machine())
    threads.append(take_photo(queue))
    threads.append(process(queue))
    threads[1].start()
    threads[0].start()
    threads[2].start()
    queue.join()
    for t in threads:
        t.join()
    stopCam(cam)
