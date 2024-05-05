#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.qos import QoSHistoryPolicy,QoSProfile,QoSReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from eufs_msgs.msg import WheelSpeedsStamped,CarState
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker


class subscriber(Node):
    def __init__(self):
        self.ac_pos_x = 0
        self.ac_pos_y = 0
        self.ac_pos_z = 0
        self.ac_pos_ori_x = 0
        self.ac_pos_ori_y = 0
        self.ac_pos_ori_z = 0
        self.ac_pos_ori_w = 0
        self.w = 0.0
        self.dt = 0.01
        self.v= 0.0
        self.u = np.array([0,0,0]).astype('float32')
        super().__init__("pose_subscriber")
        self.Imusubscriber = self.create_subscription(Imu,"/imu/data",self.imu_callback,QoSProfile(
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
      depth=QoSHistoryPolicy.UNKNOWN,
    ),)
        self.WssSubscriber = self.create_subscription(WheelSpeedsStamped,"/ros_can/wheel_speeds",self.wss_callback,10)
        self.pose_publisher = self.create_publisher(Marker,"visualize",10)
        self.pose_subscriber = self.create_subscription(CarState,"/ground_truth/state",self.pose_callback,10)
        self.actual_publisher = self.create_publisher(Marker,"actual",10)
        self.timer = self.create_timer(0.5, self.update_cylinder_pose)

    def pose_callback(self,msg:CarState):
        self.ac_pos_x = msg.pose.pose.position.x
        self.ac_pos_y = msg.pose.pose.position.y
        self.ac_pos_z = msg.pose.pose.position.z
        self.ac_pos_ori_x = msg.pose.pose.orientation.x
        self.ac_pos_ori_y = msg.pose.pose.orientation.y
        self.ac_pos_ori_z = msg.pose.pose.orientation.z
        self.ac_pos_ori_w = msg.pose.pose.orientation.w

    def imu_callback(self,msg:Imu):
        
        self.w = msg.angular_velocity.z
        
    def wss_callback(self,msg:WheelSpeedsStamped):
        
        self.v = ((msg.speeds.rb_speed + msg.speeds.lb_speed+msg.speeds.rf_speed+msg.speeds.lf_speed)/4)*0.10472*0.3
        self.u[0] += self.v*math.cos(self.u[2])*self.dt
        self.u[1] += self.v*math.sin(self.u[2])*self.dt
        
        self.u[2] = self.u[2] + self.w*self.dt
        self.get_logger().info("x: " + str(self.u[0]))
        self.get_logger().info(("y: "+str(self.u[1])))
        self.get_logger().info("theta: "+str(self.u[2]))
    
    def update_cylinder_pose(self):        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "i"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.ac_pos_x)
        marker.pose.position.y = float(self.ac_pos_y)
        marker.pose.position.z = float(self.ac_pos_z)
        marker.pose.orientation.x = float(self.ac_pos_ori_x)
        marker.pose.orientation.y = float(self.ac_pos_ori_y)
        marker.pose.orientation.z = float(self.ac_pos_ori_z)
        marker.pose.orientation.w = float(self.ac_pos_ori_w)
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.lifetime = Duration(sec=1000000)

        cy = math.cos(self.u[2] * 0.5)
        sy = math.sin(self.u[2] * 0.5)
        
        car_pose = Marker()
        car_pose.header.frame_id = 'map'
        car_pose.header.stamp = self.get_clock().now().to_msg()
        car_pose.ns = "my_namespace"
        car_pose.id = 0
        car_pose.type = Marker.CYLINDER
        car_pose.action = Marker.ADD
        car_pose.pose.position.x = float(self.u[0])
        car_pose.pose.position.y = float(self.u[1])
        car_pose.pose.position.z = 0.0
        car_pose.pose.orientation.x = 0.0
        car_pose.pose.orientation.y = 0.0
        car_pose.pose.orientation.z = float(sy)
        car_pose.pose.orientation.w = float(cy)
        car_pose.scale.x = 1.0
        car_pose.scale.y = 1.0
        car_pose.scale.z = 1.0
        car_pose.color.a = 1.0 
        car_pose.color.r = 0.0
        car_pose.color.g = 1.0
        car_pose.color.b = 0.0
        car_pose.lifetime = Duration(sec=10000000)


        self.pose_publisher.publish(car_pose)
        self.actual_publisher.publish(marker)

        

def main(args=None):

    rclpy.init(args=args)
    node = subscriber()
    rclpy.spin(node)
    rclpy.shutdown()



if __name__ == '__main__':
    main()

