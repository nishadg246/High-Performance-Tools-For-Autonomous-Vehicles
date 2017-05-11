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


def indexOfMin(array):
	indexMin = None
	for i in xrange(len(array)):
		val = array[i]
		if((indexMin == None) or (val < array[indexMin])):
			indexMin = i
	return indexMin

def indexOfMax(array):
	indexMax = None
	for i in xrange(len(array)):
		val = array[i]
		if((indexMax == None) or (val > array[indexMin])):
			indexMax = i
	return indexMax

#goes throught the array
#returns index of lowest score
#(since best correspondence has a score of 0)
def indexOfBestParticle(p_scores):
	return indexOfMin(p_scores)

#returns from 0-1
def normPositiveArray(array):
	normed = []
	maxVal = max(array)
	if(maxVal != 0.0):
		for i in xrange(len(array)):
			assert(array[i] >= 0.0)
			normed += [float(array[i])/maxVal]
	return normed

#returns from 0-1
def normSumPositiveArray(array):	
	normed = []
	if(len(array) > 0):
		maxVal = sum(array)
		if(maxVal != 0.0):
			for i in xrange(len(array)):
				assert(array[i] >= 0.0)
				normed += [round(float(array[i])/maxVal,6)]
		else:
			for i in xrange(len(array)):
				normed += [round(1.0/len(array),6)]			
		normed[0] += round(1.0 - sum(normed),6)
	return normed

def invertPositiveArray(array):
	maxVal = max(array)
	inverted = []
	for val in array:
		inverted += [maxVal - val]
	return inverted


def isValidDist(dist):
	if(len(dist.keys()) < 1):
		return False
	sumProbs = 0
	for key in dist:
		sumProbs += dist[key]
	if(abs(sumProbs - 1.0) > 0.0001):
		return False
	else:
		return True

def getGaussCounts(num_bins,precision,variance=None):
	assert(num_bins != 0)
	bins = []
	for i in xrange(num_bins):
		bins += [0]

	if(variance == None):
		variance = (float(num_bins)/6.0)
	for i in xrange(precision):
		val = int(random.gauss((float(num_bins)/2.0),variance) + 0.5)
		if((val >= 0) and (val < num_bins)):
			bins[val] += 1
	return bins

def getGaussDist(bin_size=10.0,m_range=200.0,precision=1000,variance=None):
	indices = []
	bin_size = float(bin_size)
	m_lower = float(-m_range)
	m_upper = float(m_range)
	index = m_lower
	while(index <= m_upper):
		indices += [index]
		index += bin_size

	counts = getGaussCounts(len(indices),precision,variance)
	normed = normSumPositiveArray(counts)

	bins = {}
	for i in xrange(len(indices)):
		bins[indices[i]] = normed[i]

	return bins

def sampleDist(dist):
	if(not isValidDist(dist)):
		print "sampled invalid dist: " + str(dist)
		assert(False)

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

def getHistogram(array):
	hist = {}
	for i in xrange(len(array)):
		val = array[i]
		if(val not in hist):
			hist[val] = 1
		else:
			hist[val] += 1
	return hist

def getParticleHistogram(particles):
	hist = {}
	for i in xrange(len(particles)):
		p = particles[i]
		if(p[0] not in hist):
			hist[p[0]] = 1
		else:
			hist[p[0]] += 1
	return hist

def drawParticleHistogram(histogram,fill='black',h=50,highlight=None):
	maxVal = 0
	for key in histogram:
		val = histogram[key]
		if(val > maxVal):
			maxVal = val

	for key in histogram:
		val = histogram[key]
		histogram[key] = float(val)/maxVal

	for key in histogram:

		height = h*histogram[key]
		x = key
		if(key == highlight):
			f = 'red'
		else:
			f = fill
		g.drawLine(x,0,x,-height,1,f)

def drawActualState(X):
	g.drawCircle(X[0],0,3,'red',None)



def senseObstacles(X,obs):
	depth = []
	for i in xrange(len(obs)):
		o_x = obs[i]
		x = X[0]
		depth += [o_x - x]
	return depth

def makeMap(X,depth,bin_size=10.0,m_range=200.0):
	#print "X: " + str(X)
	#print "depth: " + str(depth)
	#print "bin_size: " + str(bin_size)
	#print "m_range: " + str(m_range)



	bins = {}
	bin_size = float(bin_size)
	m_lower = float(-m_range)
	m_upper = float(m_range)
	index = m_lower
	while(index <= m_upper):
		bins[index] = 0
		index += bin_size

	for i in xrange(len(depth)):
		d_x = depth[i]
		x = X[0]



		o_x = x + d_x
		bin_x = int(float(o_x)/bin_size)*bin_size

		# print
		# print "d_x: " + str(d_x)
		# print "x: " + str(x)
		# print "o_x: " + str(o_x)
		# print "bin_x: " + str(bin_x)


		if((bin_x <= m_range) and (bin_x >= -m_range)):
			# print "added: yes"
			bins[bin_x] += 1

	# print
	# print bins
	# print
	return bins

#print makeMap([0],[-100,100],bin_size=10.0,m_range=200.0)

def compareMaps1(m0,m1):
	#print "compareMaps:"
	#print "   m0: " + str(m0)
	#print "   m1: " + str(m1)
	loss = 0
	in_m0 = set(m0.keys())
	in_m1 = set(m1.keys())
	both = in_m0.intersection(in_m1)
	either = in_m0.symmetric_difference(in_m1)
	for key in both: 
		loss += abs(m0[key] - m1[key])
	for key in either:
		if key in in_m0:
			loss += in_m0[key]
		elif key in in_m1:
			loss += in_m1[key]

	#print "   both  : " + str(both)
	#print "   either: " + str(either)
	#print "     loss: " + str(loss)


	return loss


def compareMaps(m0,m1):
	#return compareMaps1(m0,m1)
	
	score = 0

	m0_bumps = []
	m1_bumps = []
	for key in m0:
		if(m0[key] > 0):
			m0_bumps += [key]

	for key in m1:
		if(m1[key] > 0):
			m1_bumps += [key]

	for k0 in m0_bumps:
		dist = []
		for k1 in m1_bumps:
			dist += [abs(k0 - k1)]
		score += pow(min(dist),0.5)
	return score

def resampleParticlesFromScores(p_old,p_scores):
	assert(len(p_old) == len(p_scores))
	p_scores_flipped = invertPositiveArray(p_scores)
	p_prob = normSumPositiveArray(p_scores_flipped)

	p_dist = {}
	for i in xrange(len(p_scores_flipped)):
		p_dist[i] = p_prob[i]

	p_new = []
	for i in xrange(len(p_scores)):
		index = sampleDist(p_dist)
		p_new += [p_old[index]]
	return p_new

def main(num_particles, vel_actual_hist, vel_measured, vel_particle_error, obstacles):
	p_history = [] #history of particles at each timestep
	p_histogram = []
	X_history = []
	m_history = []
	s_history = []

	X_old = initState() #[0]
	p_old = initParticles(X_old, num_particles)

	#simulate sensor readings from real positions
	s_old = senseObstacles(X_old, obstacles)

	#first version of the map uses X_old
	#since all particles are initialized to this anyway
	m_old = makeMap(X_old, s_old, bin_size=5.0, m_range=250)

	X_history += [X_old]
	p_history += [p_old]
	m_history += [m_old]
	s_history += [s_old]


 	g.drawBackground(obstacles)
 	g.drawText((-SCREEN_WIDTH/2)+10,(SCREEN_HEIGHT/2)-10,str(0),10)
 	g.drawText((-SCREEN_WIDTH/2)+100,-(SCREEN_HEIGHT/2)+20,"1D Localization Simulator",15)
 	drawParticleHistogram(getParticleHistogram(p_history[0]))
 	drawActualState(X_history[0])
 	g.waitForClick()


	for t in xrange(len(vel_measured)):

	 	g.drawBackgroundRefresh(obstacles)
	 	g.drawText((-SCREEN_WIDTH/2)+10,(SCREEN_HEIGHT/2)-10,str(t),10)
	 	g.drawText((-SCREEN_WIDTH/2)+100,-(SCREEN_HEIGHT/2)+20,"1D Localization Simulator",15)

		#treat real state as a particle with zero_error
		#retrieve only the first particle in the array
		X_new = updateParticles([X_old],vel_actual_hist[t],[{0:1.0}])[0]
		X_old = X_new
		X_history += [X_new]
		print "X at t = " + str(t) + " is " + str(X_new)

		s_new = senseObstacles(X_old,obstacles)
		s_history += [s_new]
		print "s at t = " + str(t) + " is " + str(s_new)

		#get particle states updated with new velocity measurements
		p_new = updateParticles(p_old, vel_measured_hist[t], vel_particle_error)

	 	
	 	
	 	


		p_scores = []
		for i in xrange(len(p_new)):
			p_map = makeMap(p_new[i],s_new,bin_size=1.0,m_range=250)
			p_score = compareMaps(m_old,p_map)
			p_scores += [p_score]

			# p_obs = []
			# for key in p_map:
			# 	if(p_map[key] == 1):
			# 		p_obs += [key]

			#g.drawObstacles(obs=p_obs,h=40,w=2,fontSize=10,fill='gray')
			#print "map as seen by: " + str(i) + " (score: " + str(p_score) + ")"
			#print "    " + str(p_obs)
			#print "    " + str(p_new[i])
			#print "    " + str(s_new)
			

		#print "scores: " + str(getHistogram(p_scores))
		

		p_index = indexOfBestParticle(p_scores)		


		#m_new = mergeMaps(makeMap(p_new[p_index],s_new),m_old)
		p_temp = initParticles(p_new[p_index], num_particles)
		p_resampled = updateParticles(p_temp, [0], vel_particle_error)
		#p_resampled = resampleParticlesFromScores(p_new,p_scores)
		p_old = p_resampled

		#just for drawing
		p_obs = []
		p_map = makeMap(p_new[p_index],s_new,bin_size=1.0,m_range=250)
		for key in p_map:
			if(p_map[key] == 1):
				p_obs += [key]
		g.drawObstacles(obs=p_obs,h=40,w=2,fontSize=10,fill='gray')
		print "best p has state: " + str(p_new[p_index]) + " with score: " + str(p_scores[p_index])
		print "  thinks obstacles are at: " + str(p_obs)
		
		x = p_new[p_index][0]

		drawParticleHistogram(getParticleHistogram(p_new),'gray',h=50, highlight=x)
		drawParticleHistogram(getParticleHistogram(p_resampled),'black',h=-50)
		for i in xrange(50):
			drawActualState(X_new)

		g.waitForClick()
		p_history += [p_old]


error1 = {-20:0.0625,-10:0.125,0:0.625,10:0.125,20:0.0625}
error2 = {-20:0.125,-10:0.125,0:0.5,10:0.125,20:0.125}
error3 = {-20:0.125,-10:0.25,0:0.25,10:0.25,20:0.125}
error4 = {-10:0.25,0:0.5,10:0.25}
error5 = {-10:0.125,0:0.75,10:0.125}

bias1 = {-20:0.5,-10:0.25,0:0.125,10:0.125,20:0.0}

gen1 = getGaussDist(bin_size=5.0,m_range=50.0,precision=1000,variance=None)

no_error = {0:1.0}
vel_measurement_error = [bias1]
vel_particle_error    = [gen1]
#[{-0.1:0.25,0.0:0.5,0.1:0.25}]

vel_actual_hist   = [[0],[40],[-40],[20],[30],[10],[50],[-10],[-20],[-50],[0],[40],[10],[0],[-20],[0]]
vel_measured_hist = addNoiseToStates(vel_actual_hist, vel_measurement_error)

obstacles = [-200,200]

main(100, vel_actual_hist, vel_measured_hist, vel_particle_error, obstacles)