import numpy
import cv2
import numpy as np


def resize(img, num_down):
    i_h, i_w = img.shape[0:2]
    down_ratio = 2 ** num_down
    i_h_down, i_w_down = i_h // down_ratio, i_w // down_ratio
    img_down = cv2.resize(img, dsize=(i_w_down, i_h_down), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    return img_down


def img_binary(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (3, 3), 1, 1)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 4)
    return img


def findcenter(img_binary):
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 1)
    # cv2.imwrite('test2.jpg', img)
    center = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.02*length, True)
        cornerNum = len(approx)
        if length != 0:
            rate = area/length
            if rate > 30 and area > 3000 and cornerNum ==8:
                x, y, w, h = cv2.boundingRect(contours[i])
                xc = x + w/2
                yc = y + h/2
                if 0.95 < w/h < 1.05:
                    cv2.drawContours(img, contours, i, (0, 255, 0), 1)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    # cv2.circle(img, (round(xc), round(yc)), 1, (255, 0, 0), -1)
                    center.append((round(xc), round(yc)))
    # cv2.imwrite('test2.jpg', img)
    return center


def eliminate_close_center(center):
    num = len(center)
    remove = []
    for i in range(num - 1):
        if i != num - 1:
            for j in range(i+1, num):
                if abs(center[i][0] - center[j][0] + 0.1) < 3:
                    remove.append(j)
    remove = list(set(remove))[::-1]
    for m in remove:
        center.pop(m)
    return center


if __name__ == "__main__":
    img = cv2.imread('img/img02/Image_20220519142734813.bmp')

    img_binary = img_binary(img)
    center = findcenter(img_binary)
    center = eliminate_close_center(center)
    print(center)
    for c in center:
        cv2.circle(img, c, 1, (255,255,255), -1)
    cv2.imwrite('test2.jpg', img)
    cv2.imwrite('test1.jpg',img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
