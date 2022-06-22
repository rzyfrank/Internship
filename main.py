import multiprocessing
import os
import time
import sys

import threading
import cv2
import numpy as np
import math
from scipy import optimize
import functools
import matplotlib.pyplot as plt


def resize(img, num_down):
    """
    将图片按比例缩放
    :param img: 需要缩小的图片
    :param num_down: 缩小的倍数
    :return: 缩小后的图片
    """
    i_h, i_w = img.shape[0:2]
    down_ratio = 2 ** num_down
    i_h_down, i_w_down = i_h // down_ratio, i_w // down_ratio
    img_down = cv2.resize(img, dsize=(i_w_down, i_h_down), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    return img_down


# def findcontours(img):
#     contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     img_draw = np.ones(img.shape, np.uint8) * 255
#     cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 1)
#     return img_draw

#
def img_process(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 1, 1)
    # _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    # img = cv2.Canny(img, 10, 30)
    # img = findcontours(img)
    return img


def find_center(img, template, down_num):
    """
    通过模板匹配的方法找到大致的圆的中心点
    :param img: 需要检测的图片
    :param template: 用来进行模板匹配的模板，模板为需要定位的小中心圆的截图，大小方向需要尽可能与图片一致
    :param down_num: 模板匹配时需要对图片进行缩小以加快定位速度
    :return: 中心点坐标
    """

    t_h, t_w = template.shape[0:2]
    i_h, i_w = img.shape[0:2]
    down_ratio = 2 ** down_num
    t_w_down, t_h_down = t_w // down_ratio, t_h // down_ratio
    i_w_down, i_h_down = i_w // down_ratio, i_h // down_ratio
    template_down = cv2.resize(template, dsize=(t_w_down, t_h_down), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img_down = cv2.resize(img, dsize=(i_w_down, i_h_down), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    res = cv2.matchTemplate(img_down, template_down, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    centerx_down, centery_down = top_left[0] + t_w_down // 2, top_left[1] + t_h_down // 2
    centerx, centery = centerx_down * down_ratio, centery_down * down_ratio
    return centerx, centery


def max_response(response_val, type):
    """
    圆检测卡尺中，用来获取灰度值相差最大的点
    :param response_val:一个卡尺上的各点灰度值
    :param type: "w2b"为边界从白色到黑色，”b2w“为边界从黑色到白色，用于选择窄的细实线的内轮廓与外轮廓
    :return:灰度值相差最大的index
    """
    max = 0
    index = 0
    for i in range(len(response_val) - 1):
        delta = abs(response_val[i + 1] - response_val[i])
        if delta > max:
            if type == 'w2b':
                if response_val[i + 1] < response_val[i]:
                    max = delta
                    index = i
            if type == 'b2w':
                if response_val[i + 1] > response_val[i]:
                    max = delta
                    index = i
    return index


def circle_detection(img, center, r, perpendicular_len, tangential_len, angle_selection_method, type, num=None,
                     angle_selection=None):
    """

    :param img: 需要检测的图片
    :param center: 大致的圆心坐标，通过find_center的模板匹配得到
    :param r: 检测圆的大致半径
    :param perpendicular_len: 检测卡尺垂直长度的一半
    :param tangential_len: 检测卡尺水平长度的一般
    :param angle_selection_method: 检测卡尺位置安放的两种方法。第一种为”average“，表示用均分的方法来选择卡尺位置；第二种为”choose“，
                                       表示用手动选择的卡尺位置
    :param type: max_response中的参数
    :param num:如果angle_selection_method为”average“，该参数为选择多少个卡尺来平分360度的圆
    :param angle_selection: 如果angle_selection_method为”choose“, 该参数为选择参数的数组
    :return circle_contour: 每个卡尺中灰度值相差最大的点的坐标，数组
    :return contour_x: 每个卡尺中灰度值相差最大的点的x坐标，数组
    :return contour_y: 每个卡尺中灰度值相差最大的点的y坐标，数组
    """
    global angle
    pic = img

    if num is not None:
        angle = 2 * np.pi / num
    x, y = center[0], center[1]
    top_x = round(center[0] - tangential_len)
    top_y = round(center[1] - r - perpendicular_len)
    low_x = round(center[0] + tangential_len)
    low_y = round(center[1] - r + perpendicular_len)
    circle_contour = []
    contour_x = []
    contour_y = []

    if angle_selection_method == 'average':
        for i in range(num):
            mid_x, mid_y, transformed_x, transformed_y = 0, 0, 0, 0
            response_val = np.zeros(2 * perpendicular_len)
            projection_points = []
            rotate_angle = angle * i

            for row in range(top_y, low_y):
                mean_value = 0
                for col in range(top_x, low_x):
                    transformed_x = round((col - x) * math.cos(rotate_angle) - (row - y) * math.sin(rotate_angle) + x)
                    transformed_y = round((row - y) * math.cos(rotate_angle) + (col - x) * math.sin(rotate_angle) + y)

                    if col == round(top_x + tangential_len):
                        mid_x = transformed_x
                        mid_y = transformed_y

                    mean_value += img[transformed_y, transformed_x]
                mean_value /= 2 * tangential_len
                response_val[row - low_y] = mean_value
                projection_points.append((mid_x, mid_y))
            max_response_index = max_response(response_val, type)
            circle_contour.append(projection_points[max_response_index])
            contour_x.append(projection_points[max_response_index][0])
            contour_y.append(projection_points[max_response_index][1])
            cv2.circle(pic, projection_points[max_response_index], 1, (255, 0, 0), -1)
    else:
        for i in range(len(angle_selection)):
            mid_x, mid_y, transformed_x, transformed_y = 0, 0, 0, 0
            response_val = np.zeros(2 * perpendicular_len)
            projection_points = []
            rotate_angle = angle_selection[i]

            for row in range(top_y, low_y):
                mean_value = 0
                for col in range(top_x, low_x):
                    transformed_x = round((col - x) * math.cos(rotate_angle) - (row - y) * math.sin(rotate_angle) + x)
                    transformed_y = round((row - y) * math.cos(rotate_angle) + (col - x) * math.sin(rotate_angle) + y)

                    if col == round(top_x + tangential_len):
                        mid_x = transformed_x
                        mid_y = transformed_y

                    mean_value += img[transformed_y, transformed_x]

                mean_value /= (2 * tangential_len)
                response_val[row - low_y] = mean_value
                projection_points.append((mid_x, mid_y))
            max_response_index = max_response(response_val, type)
            circle_contour.append(projection_points[max_response_index])
            contour_x.append(projection_points[max_response_index][0])
            contour_y.append(projection_points[max_response_index][1])
            cv2.circle(pic, projection_points[max_response_index], 1, (255, 0, 0), -1)

    return circle_contour, contour_x, contour_y


def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


def calc_R(xc, yc):
    return np.sqrt((x_con - xc) ** 2 + (y_con - yc) ** 2)


# def calc_R1(xc, yc):
#     return np.sqrt((x_con1 - xc) ** 2 + (y_con1 - yc) ** 2)


@countcalls
def f_2(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()


# @countcalls
# def f_21(c):
#     Ri = calc_R1(*c)
#     return Ri - Ri.mean()
#

def fit_circle(x, y):
    """
    通过circle_detection得到的x,y坐标来拟合圆
    :param x: x坐标的集合，为数组
    :param y: y坐标的集合, 为数组
    :return: 拟合圆的圆心x,y坐标与半径
    """
    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, _ = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(xc_2, yc_2)
    R_2 = Ri_2.mean()

    return xc_2, yc_2, R_2


def find_circle_with_caliper(img, temp, r, perpendicular_len, tangential_len, angle_selection_method, type, num=None,
                             angle_selection=None):
    """
    圆检测函数最终接口
    :param img:
    :param temp:
    :param r:
    :param perpendicular_len:
    :param tangential_len:
    :param angle_selection_method:
    :param type:
    :param num:
    :param angle_selection:
    :return:
    """
    global x_con, y_con
    center = find_center(img, temp, 4)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 1, 1)
    # pic  = img #测试使用
    circle_con, x_con, y_con = circle_detection(img, center, r, perpendicular_len, tangential_len,
                                                angle_selection_method, type, num,
                                                angle_selection)
    x_con = np.r_[x_con]
    y_con = np.r_[y_con]
    xc, yc, r = fit_circle(x_con, y_con)
    return xc, yc, r


# def img_binary(img):
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # img = cv2.GaussianBlur(img, (3, 3), 1, 1)
#     # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2) #25,4
#     _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
#     return img


def findcenter(img):
    """
    通过寻找二值化后的轮廓，对轮廓的长度与面积进行分析，用面积长度比与面积的大小来定位目标圆的轮廓。之后用最小外接矩形的中心来表示圆的中心。
    :param img:输入的图像
    :param 函数中的筛选条件需要根据实际情况进行更改
    :return: center 中心点的xy坐标
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    center = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.02 * length, True)
        cornerNum = len(approx)
        if length != 0:
            rate = area / length
            if rate > 0.01 * length and (20000 < area < 25000 or 1500000 < area <2500000):
                x, y, w, h = cv2.boundingRect(contours[i])
                xc = x + w / 2
                yc = y + h / 2
                # cv2.drawContours(img, contours, i, (0, 255, 0), 1)
                # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                # cv2.circle(img, (round(xc), round(yc)), 1, (0, 0, 255), -1)
                center.append((round(xc), round(yc)))
    # cv2.imwrite('img/img14/test1.jpg', img)
    return center


def eliminate_close_center(center):
    """
    删除相邻的圆心。对于一些圆环或者实心圆，通过findcenter的方法一个圆会得到两个相邻的圆心，所以通过该函数进行删除后只得到一个圆心。
    :param center: findcenter函数得到的中心坐标
    :return: 消除相邻项后的圆坐标
    """
    num = len(center)
    remove = []
    for i in range(num - 1):
        if i != num - 1:
            for j in range(i + 1, num):
                if abs(center[i][0] - center[j][0] + 0.1) < 3:
                    remove.append(j)
    remove = list(set(remove))[::-1]
    for m in remove:
        center.pop(m)
    return center


def clean_bg_contours_grey(img, r, center):
    """
    只保留图形中需要的图像。
    :param img: 输入的图片
    :param r: 保留图像圆的半径
    :param center: 保留图像圆的圆心
    :return: 与img大小相同但是只有感兴趣的圆中图像的图片
    """
    h, w = img.shape
    img2 = np.ones((h, w), np.uint8) * 255
    cv2.circle(img2, (center[0], center[1]), r, (0, 0, 0), -1)
    img_copy = img.copy()
    roi = img_copy[0:h, 0:w]
    ret, mask = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img_bg, img2_fg)
    img_copy[0:h, 0:w] = dst
    return img_copy


def pick_roi_processing(img, threshold, r, center):
    """
    结合二值化与clean_bg_contours_grey
    :param img: 输入的图像
    :param threshold: 二值化的阈值
    :param r:
    :param center:
    :return: roi的黑白图像
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_roi = clean_bg_contours_grey(img_gray, r, center)
    _, img_roi = cv2.threshold(img_roi, threshold, 255, cv2.THRESH_BINARY)
    return img_roi


def calculate_area(img, center, l):
    """
    计算4个月牙黑色像素的数值（黑色面积）
    :param img: 输入的图片
    :param center: 4个月牙的中心坐标
    :param l: 可以框选4个月牙的矩形边长的一半
    :return: 四个月牙分别的面积
    """
    img1 = img[center[1] - l:center[1] + l, center[0] - l:center[0] + l]
    # img1 = resize(img1, 2)
    # cv2.imshow('img1', img1)
    # test1 = np.ones(img1.shape, np.uint8) * 255
    # test2 = np.ones(img1.shape, np.uint8) * 255
    # test3 = np.ones(img1.shape, np.uint8) * 255
    # test4 = np.ones(img1.shape, np.uint8) * 255
    area1 = 0
    area2 = 0
    area3 = 0
    area4 = 0
    a = 0
    # a = np.uint8(0)
    # print(type(a))
    # print(type(img[10,20]))
    for i in range(2 * l):
        for j in range(2 * l):
            if not img1[j, i]:
                # if img[j,i] is a:
                if (j - 2 * l) * (0 - 2 * l) - (i - 2 * l) * (0 - 2 * l) < 0 and (j - 2 * l) * (2 * l - 0) - (i - 0) * (
                        0 - 2 * l) < 0:
                    area1 += 1
                    # test1[j, i] = 0
                elif (j - 2 * l) * (0 - 2 * l) - (i - 2 * l) * (0 - 2 * l) > 0 and (j - 2 * l) * (2 * l - 0) - (
                        i - 0) * (0 - 2 * l) < 0:
                    area2 += 1
                    # test2[j, i] = 0
                elif (j - 2 * l) * (0 - 2 * l) - (i - 2 * l) * (0 - 2 * l) > 0 and (j - 2 * l) * (2 * l - 0) - (
                        i - 0) * (0 - 2 * l) > 0:
                    area3 += 1
                    # test3[j, i] = 0
                elif (j - 2 * l) * (0 - 2 * l) - (i - 2 * l) * (0 - 2 * l) < 0 and (j - 2 * l) * (2 * l - 0) - (
                        i - 0) * (0 - 2 * l) > 0:
                    area4 += 1
                    # test4[j, i] = 0

    # cv2.imwrite('img/img12/1-1.jpg', test1)
    # cv2.imwrite('img/img12/1-2.jpg', test2)
    # cv2.imwrite('img/img12/1-3.jpg', test3)
    # cv2.imwrite('img/img12/1-4.jpg', test4)
    area = [area1, area2, area3, area4]
    return area


def cal_distance(center):
    return math.sqrt((center[0][0] - center[1][0]) ** 2 + (center[0][1] - center[1][1]) ** 2)


def cal_diff(area1, area2, area3, area4):
    """
    计算月牙的面积差与偏移方向
    :param area1:
    :param area2:
    :param area3:
    :param area4:
    :return: x,y 方向上的面积差与便宜方向
    """
    x_diff = abs(area1 - area3)
    if area1 < area3:
        x_direction = "-x"
    else:
        x_direction = "+x"
    y_diff = abs(area2 - area4)
    if area2 < area4:
        y_direction = "+y"
    else:
        y_direction = "-y"

    return x_diff, y_diff, x_direction, y_direction


def area(x, r):
    S = (r ** 2) * np.arccos(x / r) - x * np.sqrt(r ** 2 - x ** 2)
    return S


def fx(n):
    global deltas, a, r
    x1 = a + n
    x2 = a - n
    return deltas - (area(x2, r) - area(x1, r))



def main(img, threshold):
    """
    凹透镜偏移检测最终接口
    :param img: 输入图片
    :param threshold: 二值化的阈值
    :param findcenter中有参数需要调整
    :return:
    """
    img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_binary, threshold, 255, cv2.THRESH_BINARY)
    center1, hierarchy1 = findcenter(img_binary)
    dis = cal_distance(center1)

    return dis


def main1(img, temp, threshold, r, param_a, param_r, scale):
    """
    凸透镜偏移检测最终接口
    :param img:
    :param temp:
    :param threshold:
    :param r:
    :param param_a:
    :param param_r:
    :param scale: 在透镜中一像素对应的距离
    :return:
    """
    a = param_a
    r = param_r
    center = find_center(img, temp, 4)
    img1 = pick_roi_processing(img, threshold, r, center)
    area1, area2, area3, area4 = calculate_area(img1, center, r+30)
    x_diff, y_diff, x_direction, y_direction = cal_diff(area1, area2, area3, area4)
    x_diff = 2000
    fx = lambda x: x_diff - (area((a-x), r) - area((a+x), r))
    fy = lambda x: y_diff - (area((a-x), r) - area((a+x), r))
    root_x = optimize.bisect(fx, 0, 30)
    root_y = optimize.bisect(fy, 0, 30)
    offset_x = root_x * scale
    offset_y = root_y * scale
    return offset_x, offset_y



# if __name__ == "__main__":
#     temp = cv2.imread('img/img06/temp.jpg')
#     distance = []
#     start1 = time.time()
#     for i in range(1, 7):
#         img = cv2.imread('img/img06/pic/'+ str(i)+ '.bmp')
#         # center = find_center(img, temp, 2)
#         # cv2.circle(img, (center[0], center[1]), 1, (0,255,0), -1)
#         # img_binary = img_binary(img)
#         start = time.time()
#         if len(img.shape) == 3:
#             img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         _, img_binary = cv2.threshold(img_binary, 30, 255, cv2.THRESH_BINARY)
#         center1, hierarchy1 = findcenter1(img_binary)
#         # print(center1)
#         dis = cal_distance(center1)
#         end = time.time()
#         print(end - start)
#         distance.append(dis)
#     end1 = time.time()
#     print(end1 - start1)
# print(distance)
# offset = [x*0.02 for x in range(1,12)]
# plt.plot(offset, distance)
# plt.xlabel('offset')
# plt.ylabel('distance')
# x_ticks = np.arange(0,0.25,0.02)
# plt.xticks(x_ticks)
# plt.savefig('img/img06/plotfig.jpg')
# plt.show()




# if __name__ == "__main__":
#     img = cv2.imread('img/img12/1.bmp')
#     temp = cv2.imread('img/img12/temp1.jpg')
#     center = find_center(img, temp, 4)
#     # cv2.circle(img, center, 220, (0,0,255), 1)
#     # cv2.imwrite('img/img10/test3.jpg', img)
#     x_different = []
#     y_different = []
#     # start = time.time()
#     for name in range(1, 9):
#         img = cv2.imread('img/img12/'+ str(name)+ '.bmp')
#         center = find_center(img, temp, 4)
#         # cv2.circle(img, center, 1, (0,255,0), -1)
#         # cv2.circle(img, center, 330, (0,0,255), 1)
#         img1 = pick_roi_processing(img, 250, 110, center)
#         area1, area2, area3, area4 = calculate_area(img1, center, 120)
#         x_diff, y_diff, x_direction, y_direction = cal_diff(area1, area2, area3, area4)
#         x_different.append(x_diff+1279)
#         y_different.append(y_diff)
#         # print(y_diff)
#         # main(img, temp)
#     # end = time.time()
#     # print(end - start)
#         # cv2.imwrite('img/img10/test/'+ str(i)+ '.jpg', img1)
#     print(x_different)
#     print(y_different)
#     offset = [x * 0.02 for x in range(0, 8)]
#     y1 = [0.0, 253.5375696863539, 506.905168493784, 759.9316429886203, 1012.4434348811374, 1264.2632761227323, 1515.2087518853878, 1765.090670704849]
#     plt.plot(offset, x_different, color='red', label='experimental value')
#     plt.plot(offset, y1, color='blue', label='theoretical value')
#     plt.xlabel('offset')
#     plt.ylabel('different')
#     x_ticks = np.arange(0,0.18,0.02)
#     plt.xticks(x_ticks)
#     plt.legend()
#     # plt.savefig('img/img10/plotfig.jpg')
#     plt.show()
#     # cv2.imwrite('img/img10/test.jpg', img1)


