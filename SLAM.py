import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
import SLAM_helper
from bresenham2D import bresenham2D

if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
 

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Doing build the first map--------')
            
        lidar_scan = self.lidar_.data_[t0]['scan'][0]
        pose = self.lidar_.data_[t0]['pose'][0]
        body_angles = self.lidar_.data_[t0]['rpy'][0]
        joint_idx = np.argmin(np.abs(self.joints_.data_['ts'][0]-self.lidar_.data_[t0]['t'][0]))
        neck_angle = self.joints_.head_angles[0][joint_idx]
        head_angle = self.joints_.head_angles[1][joint_idx]
        ray_angle = self.lidar_.range_theta_ #[dmin,dmax,last_occu,ray_angle]
        
        sx = np.ceil((pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        sy = np.ceil((pose[1]  - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

        good_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_hit = SLAM_helper.polar2cart(lidar_scan[good_range], ray_angle[good_range]) # 3*n

        world_values = SLAM_helper._ray2worldRotTrans(lidar_hit, pose, body_angles, neck_angle, head_angle)
        
        ex = np.ceil((world_values[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        ey = np.ceil((world_values[1]  - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

        good_xy = np.logical_and(np.logical_and(ex < MAP['sizex'], ex >= 0), np.logical_and(ey < MAP['sizey'], ey >= 0))
        ex = ex[good_xy]
        ey = ey[good_xy]
        
        phy_cells = bresenham2D(sx, sy, ex[0], ey[0])
        for i0 in range(len(ex)):
            if i0 > 0 and not(ex[i0] == ex[i0-1] and ey[i0] == ey[i0-1]):
                phy_cells = bresenham2D(sx, sy, ex[i0], ey[i0])

            self.log_odds_[phy_cells[0].astype(np.int32)[:-1], phy_cells[1].astype(dtype=np.int32)[:-1]] += np.log(self.p_false_)  
            self.log_odds_[int(phy_cells[0][-1]), int(phy_cells[1][-1])] += np.log(self.p_true_) 

            occ_above_thresh = self.log_odds_ > self.logodd_thresh_
            MAP['map'] = np.logical_or(MAP['map'].astype('bool'), occ_above_thresh)
            
        self.best_p_[:, t0] = pose
        self.best_p_indices_[:, t0] = [sx, sy]
        self.MAP_ = MAP



    def _predict(self,t,use_lidar_yaw=True):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))

        #using noise model
        noise = np.random.multivariate_normal([0,0,0], np.identity(3)*0.01, (self.num_p_)) #for just noise adding
        self.particles_ += noise.T

    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        MAP = self.MAP_
        corr = np.zeros(self.num_p_)
        lidar_scan = self.lidar_.data_[t]['scan'][0]
        joint_idx = np.argmin(np.abs(self.joints_.data_['ts'][0]-self.lidar_.data_[t]['t'][0]))
        body_angles = self.lidar_.data_[t]['rpy'][0]
        neck_angle = self.joints_.head_angles[0][joint_idx]
        head_angle = self.joints_.head_angles[1][joint_idx]
        ray_angle = self.lidar_.range_theta_ #[dmin,dmax,last_occu,ray_angle]

        good_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_hit = SLAM_helper.polar2cart(lidar_scan[good_range], ray_angle[good_range])

        for pnum in range(self.num_p_):
            pose = list(self.particles_[:, pnum])
           
            world_values = SLAM_helper._ray2worldRotTrans(lidar_hit, pose, body_angles, neck_angle, head_angle)
            
            ex = np.ceil((world_values[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
            ey = np.ceil((world_values[1]  - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
            
            good_xy = np.logical_and(np.logical_and(ex < MAP['sizex'], ex >= 0), np.logical_and(ey < MAP['sizey'], ey >= 0))
            ex = ex[good_xy]
            ey = ey[good_xy]
            
            corr[pnum] = np.sum(MAP['map'][ex, ey])
            
        ##### update particle weights
        log_wts = np.log(self.weights_) + corr
        max_log_wt = np.max(log_wts)
        log_wts -= (max_log_wt + prob.logSumExp(log_wts, max_log_wt))
        self.weights_ = np.exp(log_wts)

        ##### best p - update map
        best_idx = np.argmax(self.weights_)
        
        #not taking vertical transform of center of mass from ground
        sx = np.ceil((self.particles_[0, best_idx] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        sy = np.ceil((self.particles_[1, best_idx]  - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        self.best_p_indices_[:, t] = [sx, sy]
        self.best_p_[:, t] = self.particles_[:, best_idx]

        #new pose
        pose = list(self.particles_[:, best_idx])

        world_values = SLAM_helper._ray2worldRotTrans(lidar_hit, pose, body_angles, neck_angle, head_angle)

        ex = np.ceil((world_values[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        ey = np.ceil((world_values[1]  - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        good_xy = np.logical_and(np.logical_and(ex < MAP['sizex'], ex >= 0), np.logical_and(ey < MAP['sizey'], ey >= 0))
        ex = ex[good_xy]
        ey = ey[good_xy]

        phy_cells = bresenham2D(sx, sy, ex[0], ey[0])
        for i0 in range(len(ex)):
            if self.num_m_per_cell_[ex[i0], ey[i0]] < 7:
                if i0 > 0 and not(ex[i0] == ex[i0-1] and ey[i0] == ey[i0-1]):
                    phy_cells = bresenham2D(sx, sy, ex[i0], ey[i0])

                self.log_odds_[phy_cells[0].astype(np.int32)[:-2], phy_cells[1].astype(dtype=np.int32)[:-2]] += np.log(self.p_false_)  
                self.log_odds_[phy_cells[0].astype(np.int32)[-2:], phy_cells[1].astype(dtype=np.int32)[-2:]] += np.log(self.p_true_) 
                # self.log_odds_[int(phy_cells[0][-1]), int(phy_cells[1][-1])] += np.log(self.p_true_) 
                self.num_m_per_cell_[phy_cells[0].astype(np.int32), phy_cells[1].astype(np.int32)] += 1
                occ_above_thresh = self.log_odds_ > self.logodd_thresh_
                MAP['map'] = np.logical_or(MAP['map'].astype('bool'), occ_above_thresh)

        self.MAP_ = MAP
        return MAP