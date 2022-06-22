import os
import sys
import time
import math
import argparse
import copy
import json
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
import shape_based_matching_py


DEBUG_MODE = True
prefix = "/home/harry/yafei_temp/shape_based_matching-python_binding/test/"

def angle_train(src, output_dir=None, class_id="yuanzi", angle_range=[-60, 60], use_rot=True):
    if isinstance(src, np.ndarray):
        img = src
    elif isinstance(src, str):
        img = cv2.imread(src)
    else:
        print('please passing valid image path or image in np.format.')
        return

    if output_dir is not None and not Path(output_dir).exists():
        Path(output_dir).mkdir()
        
    detector = shape_based_matching_py.Detector(128, [4])

    # order of ny is row col
    #img = img[110:380, 130:400] ROI
    mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
    mask *= 255

    padding = 100
    padded_img = np.zeros((img.shape[0]+2*padding, 
        img.shape[1]+2*padding, img.shape[2]), np.uint8)
    padded_mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), np.uint8)

    padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
        img[:, :, :]
    padded_mask[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding] = \
        mask[:, :]
    # cv2.imshow("padded_img", padded_img)
    # cv2.imshow("padded_mask", padded_mask)
    # cv2.waitKey()

    shapes = shape_based_matching_py.shapeInfo_producer(padded_img, padded_mask)
    shapes.angle_range = angle_range
    shapes.angle_step = 1
    shapes.scale_range = [1]
    shapes.produce_infos()
    
    infos_have_templ = []
    is_first = True
    first_id = 0
    first_angle = 0

    for info in shapes.infos:
        to_show = shapes.src_of(info)

        templ_id = 0
        if is_first:
            templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
            first_id = templ_id
            first_angle = info.angle

            if use_rot:
                is_first = False
        else:
            templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                   info.angle-first_angle,
                shape_based_matching_py.CV_Point2f(padded_img.shape[1]/2.0, padded_img.shape[0]/2.0))
        templ = detector.getTemplates(class_id, templ_id)
        for feat in templ[0].features:
            to_show = cv2.circle(to_show, (feat.x+templ[0].tl_x, feat.y+templ[0].tl_y), 3, (0, 0, 255), -1)
        #cv2.imshow("show templ", to_show)
        #cv2.waitKey(1)
        cv2.imwrite("test/template.png", to_show)
        if templ_id != -1:
            infos_have_templ.append(info)

    if output_dir is None:
        detector.writeClasses(prefix+"%s_templ.yaml")
        shapes.save_infos(infos_have_templ, prefix + f"{class_id}_info.yaml")
    else:
        detector.writeClasses(os.path.join(output_dir, "%s_templ.yaml"))
        shapes.save_infos(infos_have_templ, os.path.join(output_dir, f"{class_id}_info.yaml"))


def angle_test(src, class_id, similarity, use_rot, templ_dir='templ_info'):
    if isinstance(src, np.ndarray):
        test_img = copy.deepcopy(src)
    elif isinstance(src, str):
        test_img = cv2.imread(src)
    else:
        print('please passing valid image path or image in np.format.')
        return None

    print('start locating...')
    detector = shape_based_matching_py.Detector(128, [4])
    ids = []
    ids.append(class_id)
    detector.readClasses(ids, os.path.join(templ_dir, "%s_templ.yaml"))

    producer = shape_based_matching_py.shapeInfo_producer()
    infos = producer.load_infos(os.path.join(templ_dir, f"{class_id}_info.yaml"))
    #test_img = cv2.imread(prefix+"case1/test.png")
    padding = 250
    padded_img = np.zeros((test_img.shape[0]+2*padding, 
        test_img.shape[1]+2*padding, test_img.shape[2]), np.uint8)
    padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
        test_img[:, :, :]

    stride = 16
    img_rows = int(padded_img.shape[0] / stride) * stride
    img_cols = int(padded_img.shape[1] / stride) * stride
    img = np.zeros((img_rows, img_cols, padded_img.shape[2]), np.uint8)
    img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
    matches = detector.match(img, similarity, ids)
    if len(matches) == 0:
        print('Template not found in test image.')
        return None
    top5 = 1
    if top5 > len(matches):
        top5 = 1
    for i in range(top5):
        match = matches[i]
        templ = detector.getTemplates(class_id, match.template_id)
        r_scaled = 1062/2.0*infos[match.template_id].scale
        train_img_half_width = 1213/2.0 + 100
        train_img_half_height = 1208/2.0 + 100
        img = cv2.circle(img, (round(match.x), round(match.y)), 3, (0, 200, 255), -1)
        x =  match.x - templ[0].tl_x + train_img_half_width
        y =  match.y - templ[0].tl_y + train_img_half_height
        print('train half width {}, height {}, tl {} {}'.format(train_img_half_width, train_img_half_height, templ[0].tl_x, templ[0].tl_y))
        print('center x, y {} {}'.format(x, y))
        img = cv2.circle(img, (round(x), round(y)), 3, (0, 255, 255), -1)
        for feat in templ[0].features:
            img = cv2.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)

        # cv2 have no RotatedRect constructor?
        print('match.template_id: {}'.format(match.template_id))
        print('match.similarity: {}'.format(match.similarity))
    
        print('center in test image {} {}'.format(x, y))
        print('image shape {}'.format(img.shape))
    print('rotating angle {}'.format(infos[match.template_id].angle))
    
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    if DEBUG_MODE:
        cv2.imwrite("test/result.png", img)
    return (round(x-250), round(y-250), infos[match.template_id].angle)

def metrology2D(img, cx, cy, r, perpendicular_len, tangential_len, rotate_angle, num=6):
    if img is None:
        print('Image is None in metraology2D.')
        return None


    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (3,3), 1.0)
    ear_interval = 60
    angle = round(rotate_angle + 360) % 360
    arc_angle = 2*math.pi/num
    first_tl_x = max(0, round(cx - tangential_len))
    first_tl_y = max(0, round(cy - r - perpendicular_len))
    rect_width = tangential_len * 2
    rect_height = perpendicular_len * 2
    circle_contour = []

    for i in range(num):
        projected_mid_x, projected_mid_y, tranformed_x, transformed_y = 0, 0, 0, 0
        response_val = np.zeros(rect_height)
        max_response_index = 0;
        projection_points = []
        arc_angle = (angle + 30 + ear_interval * i) / 180 * math.pi
        max_x = int(first_tl_x + rect_width)
        max_y = int(first_tl_y + rect_height)
        img_height, img_width = img.shape[0], img.shape[1]
        for row in range(first_tl_y, max_y):
            projected_mean = 0
            for col in range(first_tl_x, max_x):
                transformed_x = min(max(0, round((col - cx) * math.cos(arc_angle) - (cy - row) * math.sin(arc_angle) + cx)), img_width-1)
                transformed_y = round((col - cx) * math.sin(arc_angle) + (cy - row) * math.cos(arc_angle) + img.shape[0] - cy)
                transformed_y = min(img_height-1, max(0, img_height - transformed_y))
                if transformed_x < 0 or transformed_x >= img_width or transformed_y < 0 or transformed_y >= img_height:
                    print(f'transform point [{col}, {row}] to [{transformed_x} {transformed_y}] which is beyond image boundary. image width and height: {img.shape[1]} {img.shape[0]}')
                    return None

                if col == round(first_tl_x + rect_width/2):
                    projected_mid_x = transformed_x
                    projected_mid_y = transformed_y

                #cv2.circle(img, (transformed_x, transformed_y), 5, (20, 0, 0))

                projected_mean += img[transformed_y, transformed_x]
            #print('projected mean: {}, rect width: {}'.format(projected_mean, rect_width))
            projected_mean /= rect_width
            response_val[row-first_tl_y] = projected_mean
            projection_points.append((projected_mid_x, projected_mid_y))
        max_response_index = max_response(response_val)
        circle_contour.append(projection_points[max_response_index])
        cv2.circle(img, projection_points[max_response_index], 5, (20, 0, 0))

    if DEBUG_MODE:
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test/circle_points.png", color_img)
    return circle_contour


def max_response(vals):
    max_res = 0
    index = 0
    for i in range(len(vals)-1):
        vals[i] = abs(vals[i+1] - vals[i])
        if vals[i] > max_res:
            max_res = vals[i]
            index = i+1

    print('max index {}'.format(index))
    return index


def defect_gate(src_img, min_ellipse, fitting_radius, fitting_center, rotation_angle):
    if len(src_img.shape) == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    min_ellipse_center = (round(min_ellipse[0][0]), round(min_ellipse[0][1]))
    # otsu binarization
    ret, src_img = cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(src_img, 60, 60*2)
    cv2.circle(edges, min_ellipse_center, fitting_radius-50, 0, -1)
    if DEBUG_MODE:
        cv2.imwrite("test/edges.png", edges)

    conts_indices = []
    conts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i,cont in enumerate(conts):
        rect = cv2.minAreaRect(cont)
        area = rect[1][0] * rect[1][1]
        if area > 40000:
            conts_indices.append(i)
    
    edge_circle = np.zeros_like(src_img)
    for i in conts_indices:
        cv2.drawContours(edge_circle, conts, i, (255))
    if DEBUG_MODE:
        cv2.imwrite("test/edge_cleaned.png", edge_circle)

    portion_angle = 60
    angle_arc = 0
    tolerant_offset = 10
    measurement_rects = []
    left_corner, right_corner, tmp_point, mid_point = {}, {}, {}, {}
    for i in range(3):
        angle_arc = (round(portion_angle*(2*i+1) + rotation_angle - 15 + 360) % 360) / 180 * math.pi; 
        left_corner['x'] = -min_ellipse[1][0] / 2.0 * math.sin(angle_arc) + min_ellipse_center[0]
        left_corner['y'] = min_ellipse[1][0] / 2.0 * math.cos(angle_arc) + edge_circle.shape[0] - min_ellipse_center[1]
        left_corner['y'] = edge_circle.shape[0] - left_corner['y']

        angle_arc = round(portion_angle*(2*i+1) + rotation_angle) % 360; 
        if angle_arc <= 15 or angle_arc >= 345:
            mid_point['x'] = round(min_ellipse_center[0])
            mid_point['y'] = round(min_ellipse_center[1] - min_ellipse[1][0]/2.0 - tolerant_offset)
        elif angle_arc >= 75 and angle_arc <= 105:
            mid_point['x'] = round(min_ellipse_center[0] - min_ellipse[1][0]/2.0 - tolerant_offset)
            mid_point['y'] = round(min_ellipse_center[1])
        elif angle_arc >= 165 and angle_arc <= 195:
            mid_point['x'] = round(min_ellipse_center[0])
            mid_point['y'] = round(min_ellipse_center[1] + min_ellipse[1][0]/2.0 + tolerant_offset)
        elif angle_arc >= 255 and angle_arc <= 285:
            mid_point['x'] = round(min_ellipse_center[0] + min_ellipse[1][0]/2.0 + tolerant_offset)
            mid_point['y'] = round(min_ellipse_center[1])
        else:
             mid_point['x'] = -1
             mid_point['y'] = -1

        angle_arc = (portion_angle*(2*i+1) + rotation_angle + 15) / 180.0 * math.pi
        right_corner['x'] = -min_ellipse[1][0] / 2.0 * math.sin(angle_arc) + min_ellipse_center[0]
        right_corner['y'] = min_ellipse[1][0] / 2.0 * math.cos(angle_arc) + edge_circle.shape[0] - min_ellipse_center[1]
        right_corner['y'] = edge_circle.shape[0] - right_corner['y']

        if mid_point['x'] != -1 :
            if right_corner['x'] < left_corner['x'] and right_corner['x'] < mid_point['x']:
                tmp_point['x'] = right_corner['x']
            elif mid_point['x'] < left_corner['x'] and mid_point['x'] < right_corner['x']:
                tmp_point['x'] = mid_point['x']
            else:
                tmp_point['x'] = left_corner['x']

            if right_corner['y'] < left_corner['y'] and right_corner['y'] < mid_point['y']:
                tmp_point['y'] = right_corner['y']
            elif mid_point['y'] < left_corner['y'] and mid_point['y'] < right_corner['y']:
                tmp_point['y'] = mid_point['y']
            else:
                tmp_point['y'] = left_corner['y']

            if left_corner['x'] > right_corner['x'] and left_corner['x'] > mid_point['x']:
                right_corner['x'] = left_corner['x']
            elif mid_point['x'] > left_corner['x'] and mid_point['x'] > right_corner['x']:
                right_corner['x'] = mid_point['x']
            else:
                pass

            if left_corner['y'] > right_corner['y'] and left_corner['y'] > mid_point['y']:
                right_corner['x'] = left_corner['x']
            elif mid_point['y'] > left_corner['y'] and mid_point['y'] > right_corner['y']:
                right_corner['y'] = mid_point['y']
            else:
                pass
        else:
            tmp_point = left_corner
        w = abs(tmp_point['y'] - right_corner['y'])
        h = abs(tmp_point['x'] - right_corner['x'])
        measurement_rects.append({'x': round(min(tmp_point['x'], right_corner['x'])), 'y': round(min(tmp_point['y'], right_corner['y'])), 'w': round(w), 'h': round(h)})

    cv2.circle(edge_circle, min_ellipse_center, 5, (255))

    for rect in measurement_rects:
        squared_distance = 0
        for row in range(rect['y'], rect['y']+rect['h']):
            for col in range(rect['x'], rect['x']+rect['w']):
                if edge_circle[row,col] > 0:
                    squared_distance += (math.sqrt((col-min_ellipse_center[0])**2 + (row-min_ellipse_center[1])**2) - min_ellipse[1][0]/2)**2
                
        print('squared distance {}'.format(squared_distance))
        if squared_distance > 3000:
            return True

    return False


def check_gate(src_test_img, align_info, fitting_circle_radius=520, perpendicular_half_len=40, tangential_half_len=20):
    test_img = copy.deepcopy(src_test_img)
    circle_contour = metrology2D(test_img, align_info[0], align_info[1], fitting_circle_radius, perpendicular_half_len, tangential_half_len, align_info[2])
    #circle_contour = metrology2D(test_img, align_info[0]+80, align_info[1]+20, fitting_circle_radius, perpendicular_half_len, tangential_half_len, align_info[2])
    
    if circle_contour is None:
        return True
    min_ellipse = cv2.fitEllipse(np.array(circle_contour))
    test_img = cv2.ellipse(test_img, min_ellipse, (255, 0, 0))
    if DEBUG_MODE:
        cv2.imwrite("test/circled.png", test_img)
    if defect_gate(src_test_img, min_ellipse, fitting_circle_radius, (align_info[0]+80, align_info[1]+20), align_info[2]):
        print('Slicing gate NG...')
        return True
    return False


class GateProcess(mp.Process):
    def __init__(self, img_queue, out_queue):
        super(GateProcess, self).__init__()
        self.img_queue = img_queue
        self.out_queue = out_queue
    def run(self):
        if self.img_queue is None or self.out_queue is None:
            print('img_queue/out_queue is None.')
            return

        while True:
            img = self.img_queue.get(block=True)
            align_info = angle_test(img, 'painting', 90, True)
            if align_info is not None:
                ng = check_gate(img, align_info, 520, 40, 20)
                print('check ng:{}'.format(ng))
                self.out_queue.put(ng)
            
        print('exit....')

def produce(q):
    src_img = cv2.imread(prefix+"small.png")
    while True:
        q.put(src_img)
        time.sleep(1)
    



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} train|test|demo')
        exit(0)

    if sys.argv[1] == 'train':
        angle_train(sys.argv[2], sys.argv[3], "circle_3ears_smooth", angle_range=[-60, 60], use_rot=True)
    elif sys.argv[1] == 'test':
        with open('templ_info/locations.json', 'r') as f:
            infos = json.load(f)
            print(infos)
        
        info = infos['circle_3ears_smooth']
        src_img = cv2.imread(prefix+"small.png")
        align_info = angle_test(src_img, 'circle_3ears_smooth', 90, True)

        src_img = cv2.imread(prefix+"small.png")
        img = cv2.circle(src_img, (align_info[0], align_info[1]), 3, (0, 255, 0), -1)
        align_info = (align_info[0]+info['off_x'], align_info[1]+info['off_y'], align_info[2])
        print(align_info)
        img = cv2.circle(img, (align_info[0], align_info[1]), 3, (0, 0, 255), -1)
        cv2.imwrite('test/origin_point.png', img)
        if align_info is not None:
            check_gate(src_img, align_info, round(info['r']), 40, 20)
    else:
        img_queue = mp.Queue()
        output_queue = mp.Queue()
        producer = mp.Process(target=produce, args=(img_queue,))
        producer.start()
        #src_img = cv2.imread(prefix+"small.png")
        #img_queue.put(src_img)
        gate_process = GateProcess(img_queue, output_queue)
        gate_process.start()
        producer.join()
        gate_process.join()
        
        print(f'ng result: {output_queue.get()}')
        #align_info = angle_test(src_img, 'painting', 90, True)
        #if align_info is not None:
        #    check_gate(src_img, align_info, 520, 40, 20)
