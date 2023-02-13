#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
import laser_geometry.laser_geometry

laserprojection = laser_geometry.laser_geometry.LaserProjection()
pointcloud2_msg_raw = PointCloud2()


def lidar_callback(data):
    global pointcloud2_msg_raw
    #convert the message of type LaserScan to a PointCloud2
    pointcloud2_msg_raw = laserprojection.projectLaser(data)


#make equal camera coordinate and lidar coordinate
def lidar_calibration(pointcloud2_msg_raw):
    #need to edit
    calibrated_pointcloud2_msg = pointcloud2_msg_raw
    return calibrated_pointcloud2_msg


def start_lidar_calibration():
    rate = rospy.Rate(10)
    lidar_sub = rospy.Subscriber("/scan", LaserScan, lidar_callback)
    lidar_pub = rospy.Publisher("/lidar/pointcloud2", PointCloud2, queue_size=1)
    while not rospy.is_shutdown():
        rate.sleep()
        if not pointcloud2_msg_raw:
            continue
        calibrated_pointcloud2_msg = lidar_calibration(pointcloud2_msg_raw)
        lidar_pub.publish(calibrated_pointcloud2_msg)


if __name__ == '__main__':
    rospy.init_node('lidar_calibration')
    start_lidar_calibration()

