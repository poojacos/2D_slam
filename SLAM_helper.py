import numpy as np
import cv2
from scipy.special import logsumexp
import math 
import transformations as tf

def polar2cart(scan, angles):
    return np.vstack((scan * np.cos(angles), scan * np.sin(angles), np.zeros(len(scan))))

def _ray2worldRotTrans(lidar_hit, R_pose, body_angles, neck_angle, head_angle):
	lidar_hit = np.vstack((lidar_hit,np.ones((1, lidar_hit.shape[1])))) # 4*n

	head_to_lidar_trans = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.15],[0,0,0,1]]) #H_hl

	body_2_lidar_rot = np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle))
	body_2_lidar_trans = np.array([0,0,0.33])
	body_2_lidar_homo = tf.homo_transform(body_2_lidar_rot,body_2_lidar_trans) #H_bh

	world_to_part_rot = tf.twoDTransformation(body_angles[0],body_angles[1],body_angles[2])
	world_2_body_trans = np.array([R_pose[0],R_pose[1], 0.93])
	world_2_part_homo = tf.homo_transform(world_to_part_rot, world_2_body_trans) #H_gb
	
	H_gl = world_2_part_homo.dot(body_2_lidar_homo).dot(head_to_lidar_trans)
	world_hit = np.dot(H_gl, lidar_hit)

	not_floor = world_hit[2]>0
	world_hit = world_hit[:, not_floor]
	return world_hit[:2,:]
