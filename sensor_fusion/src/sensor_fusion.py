#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from mono_depth_estimation.msg import Target, Targets

points_list = []
targets_list = []


def lidar_callback(data):
    global points_list
    for point in pc2.read_points(data, field_names = ('x', 'y', 'z'), skip_nans=True):
        points_list.append([point[0], point[1]])


def target_callback(data):
    global targets_list
    targets_list = []
    for target in data.targets:
        target_detail = [target.id, target.x, target.y]
        targets_list.append(target_detail)
    print(targets_list)


#combine lidar and camera information
def sensor_fusion(points_list, targets_list):
    pass


def start_sensor_fusion():
    rate = rospy.Rate(10)
    lidar_sub = rospy.Subscriber('/lidar/pointcloud2', PointCloud2, lidar_callback)
    target_sub = rospy.Subscriber('/depth_estimation/targets', Targets, target_callback)
    while not rospy.is_shutdown():
        rate.sleep()
        if not points_list or not targets_list:
            continue
        sensor_fusion(points_list, targets_list)


if __name__ == '__main__':
    rospy.init_node('lidar_calibration')
    start_sensor_fusion()
