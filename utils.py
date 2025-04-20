import numpy as np

def phone_to_robot(accel_arr, gyro_arr):
    # Maps phone axes to robot frame: 
    # x_phone -> robot y, y_phone -> robot -x, z_phone -> robot z
    # R = np.array([[0, -1, 0],
    #               [1,  0, 0],
    #               [0,  0, 1]])
    R = np.array([[1,  0, 0],
                  [0,  1, 0],
                  [0,  0, 1]])
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
