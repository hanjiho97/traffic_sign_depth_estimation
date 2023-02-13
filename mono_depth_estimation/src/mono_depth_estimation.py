#!/usr/bin/env python3
import sys
import os
sys.path.remove(os.path.dirname(__file__))
import math
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes
from mono_depth_estimation.msg import Target, Targets

calibration_image = np.empty(shape=[0])
bbox_list_raw = []
CAMERA_HEIGHT = 0.16
TRAFFIC_SING_HEIGHT = [1, 1, 1, 1, 1]
FOV_H = 110 #need to change
FY = 346.5049
CY = 204.58251

VISUAL_IMAGE_PATH = '/home/nvidia/xycar_ws/src/mono_depth_estimation/src/car_fleid.jpg'


def image_callback(data):
    global calibration_image
    calibration_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)


def bbox_callback(data):
    global bbox_list_raw
    bbox_list_raw = []
    for bbox in data.bounding_boxes:
        bbox.xmin = max(min(639, bbox.xmin), 0)
        bbox.ymin = max(min(479, bbox.ymin), 0)
        bbox.xmax = max(min(639, bbox.xmax), 0)
        bbox.ymax = max(min(479, bbox.ymax), 0)
        box = [bbox.id, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0, 0, 0]
        bbox_list_raw.append(box)


def get_depth(bbox_list, camera_height, traffic_sign_height, fy, cy, fov_h):
    for index, bbox in enumerate(bbox_list):
        # normalized Image plane
        box_id = bbox[0]
        xmin = bbox[1]
        ymin = bbox[2]
        xmax = bbox[3]
        ymax = bbox[4]
        y_norm = (ymax - cy) / fy
        delta_x = (xmax + xmin) / 2 - 320
        azimuth = (delta_x / 320) * (fov_h / 2)
        y_distance = (1 * camera_height) / (y_norm * traffic_sign_height[box_id])
        x_distance = 100 * (y_distance * math.tan(math.pi * (azimuth/180.)))
        y_distance = 100 * (y_distance - 0.15) * 0.5826 + 2.6533
        distance = int(math.sqrt((x_distance * x_distance) + (y_distance * y_distance)))
        bbox_list[index][5] = distance
        bbox_list[index][6] = x_distance
        bbox_list[index][7] = y_distance
    return bbox_list


def write_message(bbox_list, target_results):
    for bbox in bbox_list:
        box_id = bbox[0]
        xmin = bbox[1]
        ymin = bbox[2]
        xmax = bbox[3]
        ymax = bbox[4]
        target_msg = Target()
        target_msg.x = (xmin + xmax) // 2
        target_msg.y = (ymin + ymax) // 2
        target_msg.id = box_id
        target_results.targets.append(target_msg)
    return target_results


def publish_target(bbox_list, target_pub):
    target_results = Targets()
    write_message(bbox_list, target_results)
    target_pub.publish(target_results)


def draw_box_and_depth(image, bbox_list):
    for bbox in bbox_list:
        box_id, x1, y1, x2, y2, distance, x_distance, y_distance = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{y_distance:.4}cm", (x2, y2+10), 1, 1, (255, 0, 0), 2)
    cv2.imshow('result', image)
    cv2.waitKey(1)


def draw_position(bbox_list, image_path):
    visual_fleid = cv2.imread(image_path, cv2.IMREAD_COLOR)
    visual_fleid = cv2.resize(visual_fleid, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    for bbox in bbox_list:
        x_distance, y_distance = bbox[6], bbox[7]
        cv2.circle(visual_fleid, (257+int(x_distance*5), 625-int(y_distance*4)), 10, (0, 127, 255), -1, cv2.LINE_AA)
        cv2.putText(visual_fleid, f"x:{x_distance:.4} y:{y_distance:.4}cm", (272+int(x_distance*5), 620-int(y_distance*4)), 1, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('visual_fleid', visual_fleid)
    cv2.waitKey(1)


def start_depth_estimation():
    rate = rospy.Rate(10)
    image_sub = rospy.Subscriber('/usb_cam/cailbration_image', Image, image_callback)
    bbox_sub = rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, bbox_callback)
    target_pub = rospy.Publisher('depth_estimation/targets', Targets, queue_size=1)
    while not rospy.is_shutdown():
        rate.sleep()
        if calibration_image.shape[0] == 0:
            continue
        bbox_list = get_depth(bbox_list_raw, CAMERA_HEIGHT, TRAFFIC_SING_HEIGHT, FY, CY, FOV_H)
        publish_target(bbox_list, target_pub)
        draw_box_and_depth(calibration_image, bbox_list)
        draw_position(bbox_list, VISUAL_IMAGE_PATH)


if __name__ == '__main__':
    rospy.init_node('mono_depth_estimation')
    start_depth_estimation()
