#!/usr/bin/env python3

import glob
import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from pytorchyolo import detect, models

from sensor_msgs.msg import Image
from yolov3_ros.msg import BoundingBox, BoundingBoxes

CFG_PATH = '/home/hanjiho97/xycar_ws/src/yolov3_ros/src/cfg/yolov3-tiny_tstl_416.cfg'
WEIGHTS_PATH = '/home/hanjiho97/xycar_ws/src/yolov3_ros/src/weights/model_epoch1650.weights'
YAML_PATH = '/home/hanjiho97/xycar_ws/src/yolov3_ros/src/calibration_data.yaml'
RESIZE_HEIGHT = 416
RESIZE_WIDTH = 416
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

bridge = CvBridge()
image_raw = np.empty(shape=[0])
xycar_image = np.empty(shape=[0])


class yolov3(object):
    def __init__(self):
        self.model = models.load_model(CFG_PATH, WEIGHTS_PATH)
        self.detection_publisher = rospy.Publisher('/yolov3_ros/detections', BoundingBoxes, queue_size=1)
        self.cailbration_image_publisher = rospy.Publisher('/usb_cam/cailbration_image', Image, queue_size=1)
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)
        camera_matrix, distortion_coefficients = parse_calibration_data(YAML_PATH)
        image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, image_size, 0.1, image_size)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramtx, image_size, cv2.CV_32FC1)

    def detect_box(self):
        rate = rospy.Rate(10)
        # Load the YOLO model
        while not rospy.is_shutdown():
            rate.sleep()
            if xycar_image.shape[0] == 0:
                continue

            image_raw_undistort = cv2.remap(image_raw, self.mapx, self.mapy, cv2.INTER_LINEAR)
            image_raw_undistort_resized = cv2.resize(image_raw_undistort, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            image_show = cv2.cvtColor(image_raw_undistort, cv2.COLOR_RGB2BGR)

            # Runs the YOLO model on the image
            boxes = detect.detect_image(self.model, image_raw_undistort_resized)
            # for box in boxes:
            #     x1, y1, x2, y2, confidence, class_name = box
            #     x1 = int(x1 * 640 / 416)
            #     y1 = int(y1 * 480 / 416)
            #     x2 = int(x2 * 640 / 416)
            #     y2 = int(y2 * 480 / 416)
            #     cv2.putText(image_show, f"id: {int(class_name)} confidence: {confidence:.1}", (x2, y2+10), 1, 1, (255, 0, 0), 3)
            #     cv2.rectangle(image_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.imshow('result', image_show)
            # cv2.waitKey(1)
            self.publisher(boxes, image_raw_undistort)

    def write_message(self, detection_results, boxes):
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box
            detection_msg = BoundingBox()
            detection_msg.xmin = int(x1 * 640 / 416)
            detection_msg.ymin = int(y1 * 480 / 416)
            detection_msg.xmax = int(x2 * 640 / 416)
            detection_msg.ymax = int(y2 * 480 / 416)
            detection_msg.probability = confidence
            detection_msg.id = int(class_id)
            detection_results.bounding_boxes.append(detection_msg)
        return detection_results

    def publisher(self, boxes, image):
        detection_results = BoundingBoxes()
        self.write_message(detection_results, boxes)
        self.detection_publisher.publish(detection_results)
        cailbration_image = bridge.cv2_to_imgmsg(image)
        self.cailbration_image_publisher.publish(cailbration_image)


def img_callback(data):
    global image_raw, xycar_image
    image_raw = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    xycar_image = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)


# Load the image as a numpy array
def debug_images(cfg_path, weights_path, image_folder):
    self.model = models.load_model(cfg_path, weights_path)
    image_file_path_list = glob.glob(image_folder + '*.jpg')
    for image_file_path in image_file_path_list:
        image = cv2.imread(image_file_path)

        # Convert OpenCV bgr to rgb
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (416, 416), interpolation=cv2.INTER_LINEAR)

        # Runs the YOLO model on the image
        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        boxes = detect.detect_image(self.model, image_rgb)
        for box in boxes:
            x1, y1, x2, y2, confidence, class_name = box
            x1 = int(x1 * 640 / 416)
            y1 = int(y1 * 480 / 416)
            x2 = int(x2 * 640 / 416)
            y2 = int(y2 * 480 / 416)
            cv2.putText(image, f"{class_name}", (x2, y2+10), 1, 1, (255, 0, 0), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('result', image)
        cv2.waitKey(0)


def parse_calibration_data(yaml_path):
    with open(yaml_path, 'r') as file:
        calibration_data = yaml.load(file)
    camera_matrix = np.array([[0]*3 for _ in range(3)])
    for row in range(calibration_data['CAMERA_MATRIX']['ROW']):
        for col in range(calibration_data['CAMERA_MATRIX']['COL']):
            index = row * calibration_data['CAMERA_MATRIX']['ROW'] + col
            camera_matrix[row][col] = calibration_data['CAMERA_MATRIX']['DATA'][index]
    distortion_coefficients = np.array(calibration_data['DISTORTION_COEFFICIENTS']['DATA'])
    return camera_matrix, distortion_coefficients


if __name__ == '__main__':
    yolo = yolov3()
    rospy.init_node('yolov3_ros', anonymous=True)
    yolo.detect_box()
