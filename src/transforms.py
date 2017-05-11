import numpy as np
import glob
from collections import namedtuple

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def pose_from_oxts_packet(packet):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """


    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


OxtsPacket = namedtuple('OxtsPacket','roll, pitch, yaw')
packet = OxtsPacket(np.pi/2,0,0)

#  - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
#  - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
#  - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)

pos =np.array([1.0,0.0,0.0])

#packet = OxtsPacket(*line)
#d = (times[count]-times[count-1])
#pos[0]+= packet.ve * d.microseconds / 1000000
#pos[1]+= packet.vn * d.microseconds / 1000000

delta_east = 0
delta_north = 0
delta_up = 0

pos[0] += delta_east
pos[1] += delta_north
pos[2] += delta_up

R = pose_from_oxts_packet(packet)
T_w_imu = transform_from_rot_trans(R, pos)


# velo_files = sorted(glob.glob("../../kitti/velodyne_points/data/*.bin"))
# filename = velo_files[20]

# #print "filename: " + str(filename) + "\n"
# scan = np.fromfile(filename, dtype=np.float32)

# #print "scan: " + str(scan) + "\n"
# data = scan.reshape((-1, 4))

# #print "data: " + str(data) + "\n"

# data[:,3]=1

#print "data[:,3]=1: " + str(data) + "\n"

data = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])
data[:,3]=1

#print "0th 3D point: " + str(type(data[0,:]))
print "Origin of Local Coords w.r.t. Global Origin: " + str(pos)
print "0th 3D homogenous point in Local Coords:     " + str(data[0])
print "1st 3D homogenous point in Local Coords:     " + str(data[1])
print "1st 3D homogenous point in Local Coords:     " + str(data[2])

print "0th 3D homogenous point in Global Coords: " + str((T_w_imu.dot(data[0,:].transpose())).transpose())
print "1st 3D homogenous point in Global Coords: " + str((T_w_imu.dot(data[1,:].transpose())).transpose())
print "2nd 3D homogenous point in Global Coords: " + str((T_w_imu.dot(data[2,:].transpose())).transpose())

#print "points: " + str(points) + "\n"

