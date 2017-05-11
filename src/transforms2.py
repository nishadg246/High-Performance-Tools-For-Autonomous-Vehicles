import numpy as np

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
dt = 1

v1x = np.array([2.0,0.0,0.0])
v1y = np.array([1.0,0.0,0.0])
v1z = np.array([-1.0,0.0,0.0])

roll = np.array([3.14/2,0.0,0.0])
pitch = np.array([0.0,0.0,0.0])
yaw = np.array([-3.14/2,0.0,0.0])

#lidar = [[np.array([0.0,0.0,0.0,0.0])],[np.array([-1.0,-1.0,0.0,0.0])],[np.array([-1.0,0.0,2.0,0.0])]]


origin = np.array([0.0,0.0,0.0])

p0 = origin

map_global = []

for i in xrange(len(roll)):
	Rx = rotx(roll[i])
	Ry = roty(pitch[i])
	Rz = rotz(yaw[i])
	R = Rz.dot(Ry.dot(Rx))

	t = np.array([0.0,0.0,0.0])
	R_B = R.reshape(3, 3)
	t_B = t.reshape(3, 1)
	Local_R = np.vstack((np.hstack([R_B, t_B]), [0, 0, 0, 1]))

	

	d0_1x = v1x[i] * dt 
	d0_1y = v1y[i] * dt
	d0_1z = v1z[i] * dt

	d0_1_local = np.array([d0_1x,d0_1y,d0_1z,1.0])

	d0_1_global = (Local_R.dot(d0_1_local.transpose())).transpose()
	d0_1_global = d0_1_global[0:3]

	p1 = p0 + d0_1_global

	p1_B = p1.reshape(3, 1)

	Global_RT = np.vstack((np.hstack([R_B, p1_B]), [0, 0, 0, 1]))

	print Global_RT

	quit()


	data_global = []
	for data in lidar[i]:
		data[3] = 1
		data = (Global_RT.dot(data.transpose())).transpose()
		data_global += [data[0:3]]
	map_global += [data_global]

	p0 = p1

print map_global



