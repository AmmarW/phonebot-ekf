import numpy as np

def phone_to_robot(accel_arr, gyro_arr):
    # Maps phone axes to robot frame: 
    # z_phone -> robot y, x_phone -> robot -x, y_phone -> robot z
    R = np.array([[1,  0, 0],
                  [0,  0, 1],
                  [0,  1, 0]])    # for second to third tree motion
    # R = np.array([[-1,  0, 0],
    #               [+0,  0, 1],
    #               [+0,  1, 0]])     # for straight line
      
    accel_rot = accel_arr.dot(R.T)
    gyro_rot = gyro_arr.dot(R.T)
    return accel_rot, gyro_rot

def latlon_to_enu(lat, lon, lat0, lon0):
    # Simple equirectangular approximation
    R_earth = 6378137.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R_earth * dlon * np.cos(np.radians(lat0))
    y = R_earth * dlat
    return x, y
