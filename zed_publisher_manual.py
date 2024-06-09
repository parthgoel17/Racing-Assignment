#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo,Image

class zed_publisher_manual(Node):
    def __init__(self):
        super().__init__("zed_publisher")
        self.adata_list = []
        self.cdata_list = []
        self.bdata_list = []
        self.ddata_list = []
        self.A = self.create_subscription(CameraInfo,"/zed_left/zed_node_0/left/A",self.A_callback,10)
        self.B = self.create_subscription(Image,"/zed_left/zed_node_0/left/B",self.B_callback,10)
        self.C = self.create_subscription(CameraInfo,"/zed_right/zed_node_1/left/C",self.C_callback,10)
        self.B = self.create_subscription(Image,"/zed_right/zed_node_1/left/D",self.D_callback,10)
        self.zed_publisher_A = self.create_publisher(CameraInfo,"left_0_A",10)
        self.zed_publisher_B = self.create_publisher(Image,"left_0_B",10)
        self.zed_publisher_C = self.create_publisher(CameraInfo,"left_1_C",10)
        self.zed_publisher_D = self.create_publisher(Image,"left_1_D",10)

    def A_callback(self,data1:CameraInfo):
        self.adata_list.append(data1)
        self.try_publish()
        

    def B_callback(self,data:Image):
       self.bdata_list.append(data)
       self.try_publish()
    
    def try_publish(self):
        for a in self.adata_list:
            for c in self.bdata_list:
                if abs(a.header.stamp.nanoseconds - c.header.stamp.nanoseconds) < 3e7:
                    self.zed_publisher_A.publish(a)
                    self.zed_publisher_C.publish(c)
                    if self.bdata_list[self.adata_list.index(a)]:
                        self.zed_publisher_B.publish(self.bdata_list[self.adata_list.index(a)])
                    if self.ddata_list[self.cdata_list.index(c)]:
                        self.zed_publisher_D.publish(self.ddata_list[self.cdata_list.index(c)])
                    self.adata_list.remove(a)
                    self.bdata_list.remove(self.bdata_list[self.adata_list.index(a)])
                    self.cdata_list.remove(c)
                    self.ddata_list.remove(self.ddata_list[self.cdata_list.index(c)])
                    return
        
    def C_callback(self,data:CameraInfo):
        self.cdata_list.append(data)
        self.try_publish()

    def D_callback(self,data:Image):
        self.ddata_list.append(data)

    
def main(args=None):
    rclpy.init(args=args)
    node = zed_publisher_manual()
    rclpy.spin(node) 
    rclpy.shutdown()

if __name__ == '__main__':
    main()