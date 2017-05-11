# coding: utf8
import numpy as np
import os
import glob
from collections import namedtuple
import datetime as dt
from datetime import datetime
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import mayavi.mlab


def get_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        yield scan.reshape((-1, 4))
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


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

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

velo_files = sorted(glob.glob("./velodyne_points/data/*.bin"))
oxts_files = sorted(glob.glob("./oxts/data/*.txt"))
g1 =  get_velo_scans(velo_files)
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
pos =np.array([0.0,0.0,0.0])
freq = np.zeros((1000,1000,1000))
times = []
count=0
positions = []
velo = g1.next()



points = []
raw=[]
with open("./oxts/timestamps.txt", 'r') as f:
   for line in f.readlines():
      times.append(datetime.strptime(line[:-4],"%Y-%m-%d %H:%M:%S.%f"))
for filename in oxts_files:
   with open(filename, 'r') as f:
       for line in f.readlines():
           line = line.split()
           # Last five entries are flags and counts
           line[:-5] = [float(x) for x in line[:-5]]
           line[-5:] = [int(float(x)) for x in line[-5:]]
           if (count>0 and count<len(times)):
               data = g1.next()
               data[:,3]=1
               packet = OxtsPacket(*line)
               d = (times[count]-times[count-1])
               pos[0]+= packet.ve * d.microseconds / 1000000
               pos[1]+= packet.vn * d.microseconds / 1000000
               positions.append(pos.tolist())
               R = pose_from_oxts_packet(packet)
               T_w_imu = transform_from_rot_trans(R, pos)
               points.append((T_w_imu.dot(data.transpose())).transpose())
               #points.append(data)
               #for i in data:
               #    freq[pos[0]+i[0]*4+500][pos[1]+i[1]*4+500][i[2]*4+500]+=1
           count+=1
           print count

t = np.concatenate((points[0],points[100]), axis=0)
np.random.shuffle(t)
a = freq
a [a<100]=0
x,y,z = a.nonzero()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, zdir='z', c= 'red')
plt.show()

# velopoints = [i for i in range(velo.shape[0]) if velo[i,2]>-1.7]


# def show(velo,skip):
#     velo_range = range(0, velo.shape[0], skip)
#     fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
#     mayavi.mlab.points3d(
#         velo[velo_range, 0],   # x
#         velo[velo_range, 1],   # y
#         velo[velo_range, 2],   # z
#         velo[velo_range, 2],   # Height data used for shading
#         mode="point", # How to render each point {'point', 'sphere' , 'cube' }
#         colormap='spectral',  # 'bone', 'copper',
#         #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
#         scale_factor=100,     # scale of the points
#         line_width=10,        # Scpale of the line, if any
#         figure=fig,
#     )
#     # velo[:, 3], # reflectance values
#     mayavi.mlab.show()
# show(velo[velopoints,:],1)
# fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
# plt = mayavi.mlab.points3d(
#         velo[:, 0],   # x
#         velo[:, 1],   # y
#         velo[:, 2],   # z
#         velo[:, 2],   # Height data used for shading
#         mode="point", # How to render each point {'point', 'sphere' , 'cube' }
#         colormap='spectral',  # 'bone', 'copper',
#         #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
#         scale_factor=100,     # scale of the points
#         line_width=10,        # Scpale of the line, if any
#         figure=fig,
#     )

# msplt = plt.mlab_source
# @mayavi.mlab.animate(delay=100)
# def anim():
#     while True:
#         velo = g1.next()
#         msplt.set(x=velo[:,0],y= velo[:,1], z=velo[:,2])
#         yield

# anim()
# mayavi.mlab.show()
# #show(t,1)

