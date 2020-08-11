# 2D Slam
In this project, the structure of mapping and localization in an indoor environment is implemented using information from an IMU and range sensors. Information from IMU orientation and odometry information is integrated from a walking humanoid with a 2D laser range scanner (LIDAR) in order to build a 2D occupancy grid map of the walls and obstacles in the environment. 

# Dataset 
There are two sets of data: train & test, stored in "data" repository. 
Student are given only train data. There are totally 4 map corresponding to different dataset_id (0, ..., 3)

# Run on Train dataset
```
python main.py --split_name train --dataset_id <0, 1 or 2, or 3>
```

# Generate figures 
To generate figures, run
```
python gen_figures.py --split_name <train or test>  --dataset_id <0, 1, or 2, 3> 
```

# Log files
Log file & images are all stored in "logs" repository
