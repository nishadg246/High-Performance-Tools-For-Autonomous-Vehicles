from graphics_1D import *
import random

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 200

g = graphics_1D(SCREEN_WIDTH,SCREEN_HEIGHT)

def initState():
	return [0]

def initParticle(X):
	return X

def initParticles(X,num=5):
	particles = []
	for i in xrange(num):
		particles += [initParticle(X)]
	return particles

def isValidDist(dist):
	if(len(dist.keys()) < 1):
		return False
	sumProbs = 0
	for key in dist:
		sumProbs += dist[key]
	if(sumProbs != 1.0):
		return False
	else:
		return True

#def genGaussDist(variance,bins,radius):
#	binSize = (radius*2)/bins
#	for i in xrange(bins)

def sampleDist(dist):
	assert(isValidDist(dist))
	boundaries = []
	output = []
	runningSum = 0
	for i in xrange(len(dist.keys())):
		key = dist.keys()[i]
		val = dist[key]
		runningSum += val
		boundaries += [runningSum]
		output += [key]

	sample_val = random.random()
	for i in xrange(len(boundaries)):
		if(sample_val <= boundaries[i]):
			return output[i]
	assert(False)
	return output[-1]

def addNoiseToState(val,dist):
	error = sampleDist(dist)
	return val + error

def addNoiseToStates(X0s,dist):
	X1s = []
	for i in xrange(len(X0s)):
		X0 = X0s[i]
		X1 = []
		for j in xrange(len(X0)):
			X1 += [addNoiseToState(X0[j],dist[j])]
		X1s += [X1]
	return X1s

# updated


# val = addNoise(p_old[i] + vel_measured[i],vel_dist[i])
# vel_error = sampleDist(vel_dist[i])
# val += vel_measured[i] + vel_error

def updateParticle(p_old, vel_measured):
	p_new = []
	for i in xrange(len(p_old)):
		p_new += [p_old[i] + vel_measured[i]]
	return p_new

def updateParticles(p_old, vel_measured, vel_particle_error):
	p_temp = []
	for a in p_old:
		b = updateParticle(a, vel_measured)
		p_temp += [b]
	p_new = addNoiseToStates(p_temp, vel_particle_error)
	return p_new

def drawParticle(p):
	g.drawCircle(p[0],0,1,None,'red')

def drawParticles(particles):
	for p in particles:
		drawParticle(p)

def getParticleHistogram(particles):
	hist = {}
	for i in xrange(len(particles)):
		p = particles[i]
		if(p[0] not in hist):
			hist[p[0]] = 1
		else:
			hist[p[0]] += 1
	return hist

def main(num_particles, vel_actual_hist, vel_measured, vel_particle_error, obstacles):

	

	p_hist = []
	#X_hist = []

	X = initState()
	p_old = initParticles(X,num_particles)

	#sense(X_new, obstacles)
	p_hist += [p_old]

	for i in xrange(len(vel_measured)):
		#vel_actual   = vel_actual_hist[i]
		vel_measured = vel_measured_hist[i]


		#X = updateParticles([X],vel_actual_hist[i],{0:1.0})
		p_new = updateParticles(p_old, vel_measured, vel_particle_error)
		p_hist += [p_new]
		p_old = p_new

		#X_hist += [X]

	for i in xrange(len(p_hist)):
		p = p_hist[i]

	 	g.drawBackground(obstacles)
	 	g.drawText((-SCREEN_WIDTH/2)+10,(SCREEN_HEIGHT/2)-10,str(i),10)
	 	#drawParticles(p)
	 	g.waitForClick()

	print p_hist
	#print X_hist


dist = {-20:0.125,-10:0.25,0.0:0.25,10:0.25,20:0.125}
vel_measurement_error = [dist]
vel_particle_error    = [dist]
#[{-0.1:0.25,0.0:0.5,0.1:0.25}]

vel_actual_hist   = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
vel_measured_hist = addNoiseToStates(vel_actual_hist,vel_measurement_error)

obstacles = [-200,200]

#main(100, vel_actual_hist, vel_measured_hist, vel_particle_error, obstacles)

print getParticleHistogram([[0],[1],[2],[3],[4],[4]])
