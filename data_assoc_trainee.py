#!/usr/bin/env python3

'''
motion update for mrpt-fsds 
'''
from utility import *

import rospy
import tf2_ros
import numpy as np
import message_filters
from math import sin,cos,tan,atan2,sqrt,pi, exp, hypot

from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, Pose, TransformStamped
from fs_msgs.msg import Track
from nav_msgs.msg import Odometry
from mrpt_msgs.msg import ObservationRangeBearing, SingleRangeBearingObservation
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float64

# global_car_x = 0
# global_car_y = 0
# global_car_theta = 0


class Estimator():
    '''
    Description in short pls
    '''
    # carPose = Pose()  ## using mu_t to store cars exact position
    carPoseCov = np.identity(3, dtype='float')
    mu_t = np.array([[0],[0],[0]])

    odomSeq = 0
    dt = 0.004

    id = 0
    
    ## parameters for measurement update
    # ConesGlobalTrack = Track()  ## mu_t should store global cones map 
    cones_distance_threshold = 1.5
    N : int  = 0 ## number of cones stored in the map

    # for error covariarance in motion_update
    v_sigma = 0.25
    w_sigma = 1e-5
    range_sigma = 0.1
    angle_sigma = 0.0349/2
    id_sigma = 0.01
    Qt = np.array([[range_sigma, 0, 0], 
                   [0, angle_sigma, 0], 
                   [0, 0, id_sigma]], dtype='float')
    car_x_motionUpdate_only = 0.0
    car_y_motionUpdate_only = 0.0
    car_theta_motionUpdate_only = 0.0



    odomSeq = 0

    updateValuesAfterMeasUpdate = True
    measurement_calculations_goingon = False

    def __init__(self) -> None:
        pass

    

    def store_gss_data(self, data):
        self.gss_data = data



        



    def motion_update(self, fromIMU):
        if self.measurement_calculations_goingon == True:
            pass

        if self.updateValuesAfterMeasUpdate == False: ## values need to be updateed
            # print("valaues updated after measurement update before starting next motion update")
            self.mu_t = self.mu_t_afer_meas
            self.carPoseCov = self.carPoseCov_after_meas
            self.N = self.N_afer_meas
            self.updateValuesAfterMeasUpdate = True

            self.measurement_calculations_goingon = False


        # print("motion update started and N is ",  self.N)
        # print(" state matrix = " , self.mu_t)
        fromGSS = self.gss_data
        v = sqrt(fromGSS.twist.twist.linear.x**2 + fromGSS.twist.twist.linear.y**2)
        w = fromIMU.angular_velocity.z
        phi_v = self.mu_t[2, 0]
        yaw = self.mu_t[2, 0]

        # vx = v * cos(phi_v)
        # vy = v * sin(phi_v)
        vx = fromGSS.twist.twist.linear.x
        vy = fromGSS.twist.twist.linear.y

        dx = (v*sin(yaw + w*self.dt)/w) - (v*sin(yaw)/w)
        dy = (v*cos(yaw)/w) - (v*cos(yaw + w*self.dt)/w)
        dyaw = w*self.dt

        # print('dx is ' ,dx , 'dy is ', dy ,'and dyaw is ', dyaw)

        # G_t = np.array([[1, 0, (vx * sin(phi_v) - vy * cos(phi_v)) * self.dt],
        #                 [0, 1, (vx * cos(phi_v) - vy * sin(phi_v)) * self.dt],
        #                 [0, 0, self.dt]], dtype = 'float')

        G_t = np.array([[1, 0 , dy],
                        [0 , 1 , -dx],
                        [0, 0 ,1 ]], dtype = 'float')  ## the last row self.dt can be changed with 1 as in the previous code

        Q_t_l = np.array([[self.v_sigma, 0, 0],
                            [0, self.v_sigma, 0],
                            [0, 0, self.w_sigma]], dtype = 'float')

        V_t = np.array([[cos(phi_v) * self.dt, - sin(phi_v) * self.dt, 0],
                        [sin(phi_v) * self.dt, cos(phi_v) * self.dt, 0],
                        [0, 0, 1]], dtype = 'float')

        Q_t = np.matmul(V_t , np.matmul(Q_t_l , V_t.T))


        ## for just chcking if measuremenent update im improving motion update ornot
        self.car_x_motionUpdate_only = self.car_x_motionUpdate_only + dx
        self.car_y_motionUpdate_only = self.car_y_motionUpdate_only + dy
        self.car_theta_motionUpdate_only = self.car_theta_motionUpdate_only + dyaw


        # print(self.car_x_motionUpdate_only, self.car_y_motionUpdate_only)


        

        ###   USIGN GLOBAL VARIABLES ABOVE INSTEAD OF MU_T AS IT COULD INTERRUPT WITH THE MEASUREMENT UPDATE

        if self.measurement_calculations_goingon == True:
            pass

        mu_t_v_pred = np.array([[self.mu_t[0,0] + dx],
                                [self.mu_t[1, 0] + dy],
                                [self.mu_t[2, 0] + dyaw]])



        # print(mu_t_v_pred)

        # print('N is ', self.N)
        if(self.N != 0):
            last_2n_elements = self.mu_t[3:,:]
            mu_t_v_pred = np.vstack((mu_t_v_pred, last_2n_elements))

        ## regarding covariances
        if(self.N == 0):
            top_left = np.matmul(G_t , np.matmul(self.carPoseCov , G_t.T))
            sigma_t_pred = top_left

        else:
            top_left = np.matmul(G_t, np.matmul(self.carPoseCov[:3,:3], G_t.T))
            top_right = np.matmul(G_t,self.carPoseCov[:3, 3: ])
            bottom_left = (np.matmul(G_t, self.carPoseCov[:3, 3: ])).T
            bottom_right = self.carPoseCov[3:,3:]
            # print("top_left = " , top_left.shape , "top_right = " ,top_right.shape , "bottom_left =", bottom_left.shape, "bottom_right = ", bottom_right.shape)
            top = np.concatenate((top_left, top_right), axis = 1)
            bottom = np.concatenate((bottom_left, bottom_right), axis = 1)
            sigma_t_pred = np.concatenate((top, bottom), axis = 0)
            sigma_t_pred = sigma_t_pred  #   + Q_t ## have to lookfor a way to Qrt

        # print(self.mu_t)
        # print(mu_t_v_pred)
        self.mu_t = mu_t_v_pred
        # self.mu_t[0,0] = mu_t_v_pred[0,0]
        # self.mu_t[1,0]  = mu_t_v_pred[1, 0]
        # self.mu_t[2,0] = mu_t_v_pred[2,0]
        # print(self.mu_t)
        self.carPoseCov = sigma_t_pred

        # print("predicted car_x = ", self.mu_t[0,0] , " and car_y = " ,self.mu_t[1,0])

        # print('yaw rate is ', w, ' and orientation of car is' , self.mu_t[2,0])
        '''
        Publishing odomMarker GREEN-cubes
        '''
        pred_pose = Pose()
        pred_pose.position.x = self.mu_t[0,0]
        pred_pose.position.y = self.mu_t[1,0]
        PoseMarkerMsg = Marker()
        PoseMarkerMsg.header.frame_id = 'map'
        PoseMarkerMsg.ns = "testing_only"
        # PoseMarker.id = self.odomSeq
        PoseMarkerMsg.type = 0
        PoseMarkerMsg.action = 0
        PoseMarkerMsg.pose = pred_pose
        
        PoseMarkerMsg.scale.x = 0.1
        PoseMarkerMsg.scale.y = 0.1 
        PoseMarkerMsg.scale.z = 0.1 
        PoseMarkerMsg.color.r = 0
        PoseMarkerMsg.color.g = 256
        PoseMarkerMsg.color.b = 256
        PoseMarkerMsg.color.a = 1
        PoseMarkerMsg.lifetime = rospy.Duration(0.01)        
        PoseMarker.publish(PoseMarkerMsg)

        self.odomSeq += 1
        # print("motion update ended")

        # self.measurement_update()

        # return mu_t_v_pred, sigma_t_pred

        # global_car_x = self.mu_t[0,0]
        # global_car_y = self.mu_t[1,0]
        # global_car_theta = self.mu_t[2,0]

    def measurement_update(self, data):

        cones_from_perception = data.track
        N_old = self.N
        mu_t_old  = self.mu_t
        carPoseCov_old = self.carPoseCov
        # car_x_old = global_car_x
        # car_y_old = global_car_y
        # car_theta_old = global_car_theta

        car_x_old = mu_t_old[0, 0]
        car_y_old = mu_t_old[1, 0]
        car_theta_old = mu_t_old[2, 0]

        print("measurement update started and N is " ,N_old)
        num_perception_cones = len(cones_from_perception)

        # print("number of perception cones is = " , num_perception_cones , "")

        i = 0 ## number of matched measurements

        perception_cone_mapped = []        ## corresponding cone from perception mapped with which cone from original map, if not matched -1
        for cone_perception in cones_from_perception:
            cone_match = self.basic_data_association(cone_perception, mu_t_old, N_old)
            if(cone_match != -1):
                i = i + 1
            perception_cone_mapped.append(cone_match)  ## if the cone is mapped, index is from (0 to n-1) where n are the total cones in the map

        N_new , mu_t_old , carPoseCov_old =  self.initialising_new_landmarks(N_old, perception_cone_mapped, cones_from_perception , mu_t_old, carPoseCov_old)

        print("number of perception cones is = " , num_perception_cones , "matched cones = ", i , "newN = " , N_new)

        H_t = np.zeros((2 * i , N_new * 2 + 3))
        del_z_t = np.zeros((2 * i , 1))
        R_t = np.zeros((2 * i , 2 * i))

        global_index_of_matched_cones = [] ## indexes of the global cones that have been associated with the incoming cones
        cones_from_perception_matched = []
        for perception_cone_index, if_cone_mapped in enumerate(perception_cone_mapped):
            if if_cone_mapped != -1:
                global_index_of_matched_cones.append(if_cone_mapped)
                cones_from_perception_matched.append(cones_from_perception[perception_cone_index])

        for j in range(0, len(cones_from_perception_matched)):
            del_x = mu_t_old[3 + 2 * global_index_of_matched_cones[j], 0] - mu_t_old[0, 0]
            del_y = mu_t_old[3 + 2 * global_index_of_matched_cones[j] + 1, 0] - mu_t_old[1, 0]
            
            q = del_x ** 2 + del_y ** 2

            cone_from_perception = cones_from_perception_matched[j]
            range_cone = cone_from_perception.location.x
            pitch = cone_from_perception.location.y

            z_j_t = np.array([[range_cone],
                            [pitch]])

            pred_z_j_t = np.array([[sqrt(q)],
                                    [atan2(del_y, del_x)]])

            del_z_t[2 * j, 0] =  z_j_t[0, 0] - pred_z_j_t[0, 0]
            del_z_t[2 * j + 1, 0] = z_j_t[1, 0] - pred_z_j_t[1, 0]

            H_j_t_v = np.array([[-sqrt(q) * del_x , -sqrt(q) * del_y , 0],
                                [      del_y     ,     - del_x   , - q]])

            H_j_t_j = np.array([[sqrt(q) * del_x , sqrt(q) * del_y],
                                [ - del_y     ,     del_x   ]])

            self.modify_matrix(H_t , 2 * j , 0 , 2 , 3 , H_j_t_v)
            self.modify_matrix(H_t , 2 * j ,  3 + 2 * j , 2 , 2 , H_j_t_j)

            sig_r = np.array([[self.range_sigma]])
            sig_theta = np.array([[self.angle_sigma]])

            self.modify_matrix(R_t , 2 * j , 2 * j , 1 , 1 , sig_r)
            self.modify_matrix(R_t, 2 * j + 1, 2 * j + 1, 1 , 1 , sig_theta)


        # print(H_t.shape)
        # print(carPoseCov_old.shape)
        # print(R_t.shape)
        YYY = np.linalg.inv( np.matmul(H_t , np.matmul(carPoseCov_old , H_t.T)) + R_t )


        K_t = np.matmul(carPoseCov_old , np.matmul( H_t.T , YYY))



        mu_t_new = mu_t_old + np.matmul( K_t , del_z_t)

        carPoseCov_new = np.matmul( np.eye(2 * N_new + 3) - np.matmul(K_t, H_t) , carPoseCov_old)

        # car_x_new = global_car_x
        # car_y_new = global_car_y
        # car_theta_new = global_car_theta

        # ## adding values in mu_t for making it parallel
        # self.mu_t[0,0] = self.mu_t[0,0] + car_x_new - car_x_old
        # self.mu_t[1,0] = self.mu_t[1,0] + car_y_new - car_y_old
        # self.mu_t[2,0] = self.mu_t[2,0] + car_theta_new - car_theta_old

        # global_car_x = self.mu_t[0,0]
        # global_car_y = self.mu_t[1,0]
        # global_car_theta = self.mu_t[2,0]



        # self.print_pred_cones()
        self.print_new_cones(mu_t_new,  N_old, N_new)


        self.measurement_calculations_goingon = True
        print("measurement update ended and N is " ,self.N)


        self.adjust_pose_after_measurement(car_x_old, car_y_old, car_theta_old, N_new, mu_t_new, carPoseCov_new)




    def adjust_pose_after_measurement(self, car_x_old, car_y_old, car_theta_old , N_new, mu_t_new, carPoseCov_new):
        curr_state = self.mu_t
        mu_t_new[0,0] = mu_t_new[0,0] + curr_state[0,0] - car_x_old
        mu_t_new[1, 0] = mu_t_new[1, 0] + curr_state[1, 0] - car_y_old
        mu_t_new[2, 0 ] = mu_t_new[2, 0] + curr_state[2, 0] - car_theta_old

        self.mu_t_afer_meas = mu_t_new
        self.carPoseCov_after_meas = carPoseCov_new
        self.N_afer_meas = N_new
        self.updateValuesAfterMeasUpdate = False ## false means the values are not updated and needed to be updated






            
    def initialising_new_landmarks(self, N_old, perception_cone_mapped, cones_from_perception,  mu_t, sigma_t):
        # car_x = self.carPose.position.x
        # car_y = self.carPose.position.y
        # q = self.carPose.orientation
        # car_angle = atan2(2.0*(q.w*q.z + q.x*q.y), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

        mu_t_old = mu_t

        car_x = mu_t_old[0,0]
        car_y = mu_t_old[1, 0]
        car_angle = mu_t_old[2, 0]

        print("starting initialising_new_landmarks and N is =" , self.N)

        for i, if_cone_mapped in enumerate(perception_cone_mapped):
            if if_cone_mapped == -1:
                cone_to_be_added = cones_from_perception[i]
                range_cone = cone_to_be_added.location.x
                pitch = cone_to_be_added.location.y
                mu_ix = car_x + range_cone * cos(car_angle + pitch)
                mu_iy = car_y + range_cone * sin(car_angle + pitch)
                mu_t = np.vstack((mu_t, mu_ix, mu_iy))
                H_v_inv = np.array([[1, 0, -range_cone * sin(car_angle + pitch)]
                                    , [0, 1, range_cone * sin(car_angle + pitch)]])
                R_i_t = np.array([[self.range_sigma, 0],
                                   [ 0, self.angle_sigma]])
                H_inv = np.zeros((2, 2 * N_old + 3))
                ## H_inv bblock(0, 0, 3,3) -- left to write
                self.modify_matrix(H_inv, 0, 0, 2, 3, H_v_inv)
                H_inv_i = np.array([[cos(car_angle + pitch), -range_cone * sin(car_angle + pitch)],
                                    [sin(car_angle + pitch), range_cone * cos(car_angle + pitch)]])

                # print(sigma_t.shape,  H_inv.shape)


                top_right = np.matmul(sigma_t.T, H_inv.T) ##sigma_t.T.matmul(H_inv) 
                bottom_left = np.matmul(H_inv, sigma_t) ##H_inv.matmul(sigma_t)
                # bottom_right = H_inv.matmul(sigma_t.matmul(H_inv.T)) + H_inv_i.matmul(R_i_t.matmul(H_inv_i.T))
                bottom_right = np.matmul(H_inv , np.matmul(sigma_t , H_inv.T)) + np.matmul(H_inv_i , np.matmul(R_i_t , H_inv_i.T))
                top = np.concatenate((sigma_t,  top_right), axis = 1)
                bottom = np.concatenate((bottom_left, bottom_right), axis = 1 )
                sigma_t = np.concatenate((top, bottom), axis = 0)
                N_old = N_old + 1
                # print("cone added")

        print("initialising_new_landmarks ended and n is ", self.N)
        return N_old,  mu_t, sigma_t

    

    def vizualize_truePose(self, data):
        '''
        Publishing odomMarker GREEN-cubes
        '''
        truePoseMarkerMsg = Marker()
        truePoseMarkerMsg.header.frame_id = 'map'
        truePoseMarkerMsg.ns = "testing_only"
        # truePoseMarker.id = self.odomSeq
        truePoseMarkerMsg.type = 0
        truePoseMarkerMsg.action = 0
        truePoseMarkerMsg.pose = data.pose.pose
        
        truePoseMarkerMsg.scale.x = 0.1
        truePoseMarkerMsg.scale.y = 0.1 
        truePoseMarkerMsg.scale.z = 0.1 
        truePoseMarkerMsg.color.r = 0
        truePoseMarkerMsg.color.g = 256
        truePoseMarkerMsg.color.b = 0
        truePoseMarkerMsg.color.a = 1
        truePoseMarkerMsg.lifetime = rospy.Duration(0.01)        
        truePoseMarker.publish(truePoseMarkerMsg)

        # print(" original car_x = ", data.pose.pose.position.x , "and car_y = ", data.pose.pose.position.y)
        
        self.odomSeq += 1


        error_meas = Float64()
        error_motion = Float64()

        error_meas.data = (data.pose.pose.position.x - self.mu_t[0,0])**2 + (data.pose.pose.position.y - self.mu_t[1,0])**2
        error_motion.data = (data.pose.pose.position.x - self.car_x_motionUpdate_only) ** 2 + (data.pose.pose.position.y - self.car_y_motionUpdate_only) ** 2

        error_carPose_measurement.publish(error_meas)
        error_carPose_motion.publish(error_motion)


    
    def basic_data_association(self,cone_perception, mu_t_old, N_old):
        
        threshold = 1000
        min = 1000000
        D = ((mu_t_old[0] - cone_perception.location.x)**2 + (mu_t_old[1] - cone_perception.location.y)**2)**0.5
        for i in range(N_old):
            d = ((mu_t_old[0] - mu_t_old[(2*i)+3])**2 + (mu_t_old[1] - mu_t_old[(2*i)+4])**2)**0.5
            if d<min:
                min = d
        if abs(min-D) <threshold:
            return cone_perception
        return -1
            

        
    def modify_matrix(self, matrix, begin_row, begin_col, num_rows, num_cols, matrix_to_be_put_in_there):
        assert matrix_to_be_put_in_there.shape == (num_rows, num_cols), f"matrix that is to be put there has not the required shape of({num_rows}, {num_cols})"
        for row in range(begin_row, begin_row + num_rows):
            for col in range(begin_col, begin_col + num_cols):
                matrix[row][col] = matrix_to_be_put_in_there[row - begin_row, col - begin_col]
        
    def distance_between_cone(self, cone_mapped, cone_x, cone_y):
        return math.sqrt((cone_x - cone_mapped.location.x) ** 2 + (cone_y - cone_mapped.location.y) ** 2)
        

        
    def print_pred_cones(self):
        cones = MarkerArray()
        for i in range(0, self.N):
            marker = Marker()
            marker.ns = "cones_predected"
            marker.type = 2
            marker.action = 0
            
            marker.id = self.id
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 1
            marker.color.a = 0.5
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            marker.pose.position.x = self.mu_t[3 + 2 * i , 0]
            marker.pose.position.y = self.mu_t[3 + 2 * i + 1, 0]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.lifetime = rospy.Duration(0)
            marker.header.frame_id = 'map'
            cones.markers.append(marker)
            self.id += 1

        pred_cones.publish(cones)


    def print_new_cones(self,mu_t,  N_old, N_new):
        cones = MarkerArray()
        for i in range(N_old, N_new):
            marker = Marker()
            marker.ns = "cones_predected"
            marker.type = 2
            marker.action = 0
            
            marker.id = self.id
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 1
            marker.color.a = 0.5
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            marker.pose.position.x = mu_t[3 + 2 * i , 0]
            marker.pose.position.y = mu_t[3 + 2 * i + 1, 0]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.lifetime = rospy.Duration(0)
            marker.header.frame_id = 'map'
            cones.markers.append(marker)
            self.id += 1

        print_new_pred_cones.publish(cones)


if __name__=='__main__':
    rospy.init_node('motion_update')

    E = Estimator()

   


    # For Motion update (time synchronized subscribing to IMU and GSS)
    # fromIMU = message_filters.Subscriber('/fsds/imu', Imu)
    # fromGSS = message_filters.Subscriber('/fsds/gss', TwistWithCovarianceStamped)
    # toMotion = message_filters.ApproximateTimeSynchronizer([fromGSS, fromIMU], 10, 0.003)
    # toMotion.registerCallback(E.motion_update)

    

    rospy.Subscriber('/fsds/imu', Imu, E.motion_update)
    rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, E.store_gss_data)

  

    rospy.Subscriber('/ConesSeen', Track, E.measurement_update)

    # rospy.Subscriber('/fsds/testing_only/odom', Odometry, E.motion_update_exactposeFromFSDS)
              

    rospy.Subscriber('/fsds/testing_only/odom', Odometry, E.vizualize_truePose)




    truePoseMarker = rospy.Publisher('truePose_viz', Marker, queue_size=10)

    PoseMarker = rospy.Publisher("pred_pose", Marker, queue_size = 10)

    pred_cones = rospy.Publisher("pred_cones", MarkerArray, queue_size = 10)
    print_new_pred_cones = rospy.Publisher("new_cones", MarkerArray, queue_size = 10)

    error_carPose_measurement = rospy.Publisher("error_carPose_measurement", Float64 ,  queue_size = 10 )
    error_carPose_motion = rospy.Publisher("error_carPose_motion", Float64 , queue_size = 10 )


    # Publishing Odometry transfo
    tfodom = tf2_ros.TransformBroadcaster()

    # Publishing final car pose from parallel structure
    # CarPose = rospy.Publisher('parallelCarPose' , Pose, queue_size=10)


    # motionUpdatePos = rospy.Publisher('botPose', Marker, queue_size=10)

    rospy.spin()