import cv2
from CamUse_import1 import *

deviceList = enum_devices(device=0, device_way=False)
identify_different_devices(deviceList)
nConnectionNum = 0
cam, stDeviceList = creat_camera(deviceList, nConnectionNum)
nPayloadSize = open_device(cam, stDeviceList)
start_grabing(cam)
start = time.time()
img1 = access_get_image(cam, nPayloadSize)
img2 = access_get_image(cam, nPayloadSize)
end = time.time()
print(end - start)
stop_grabing(cam)
close_device(cam)
destroy_handle(cam)

img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)