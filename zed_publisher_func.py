#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo,Image

class zed_publisher_func(Node):
    def __init__(self):
        super().__init__("zed_publisher")
        self.A = Subscriber(self,CameraInfo,"/zed_left/zed_node_0/left/A")
        self.B = Subscriber(self,Image,"/zed_left/zed_node_0/left/B")
        self.C = Subscriber(self,CameraInfo,"/zed_right/zed_node_1/left/C")
        self.D = Subscriber(self,Image,"/zed_right/zed_node_1/left/D")

        ats = ApproximateTimeSynchronizer([self.A, self.B,self.C,self.D], queue_size=10, slop=0.1,allow_headerless=True)
        ats.registerCallback(self.synced_callback)
        self.zed_publisher_A = self.create_publisher(CameraInfo,"left_0_A",10)
        self.zed_publisher_B = self.create_publisher(Image,"left_0_B",10)
        self.zed_publisher_C = self.create_publisher(CameraInfo,"left_1_C",10)
        self.zed_publisher_D = self.create_publisher(Image,"left_1_D",10)

    def synced_callback(self,msg_a,msg_b,msg_c,msg_d):
        self.zed_publisher_A.publish(msg_a)
        self.zed_publisher_B.publish(msg_b)
        self.zed_publisher_C.publish(msg_c)
        self.zed_publisher_D.publish(msg_d)
    
def main(args=None):
    rclpy.init(args=args)
    node = zed_publisher_func()
    rclpy.spin(node) 
    rclpy.shutdown()

if __name__ == '__main__':
    main()