#from graphics_1D import *
import random


# W = 800
# H = 400

# g = graphics_1D(W,H)

# #returns from 0-1
# def normSumPositiveArray(array):	
# 	normed = []
# 	if(len(array) > 0):
# 		maxVal = sum(array)
# 		if(maxVal != 0.0):
# 			for i in xrange(len(array)):
# 				assert(array[i] >= 0.0)
# 				normed += [round(float(array[i])/maxVal,6)]
# 		else:
# 			for i in xrange(len(array)):
# 				normed += [round(1.0/len(array),6)]			
# 		normed[0] += round(1.0 - sum(normed),6)
# 	return normed

# def getGaussCounts(num_bins,precision,variance=None):
# 	assert(num_bins != 0)
# 	bins = []
# 	for i in xrange(num_bins):
# 		bins += [0]

# 	if(variance == None):
# 		variance = (float(num_bins)/6.0)
# 	for i in xrange(precision):
# 		val = int(random.gauss((float(num_bins)/2.0),variance) + 0.5)
# 		if((val >= 0) and (val < num_bins)):
# 			bins[val] += 1
# 	return bins

# def getGaussDist(points=10,precision=1000,variance=None):
# 	indices = []
# 	for i in xrange(points):
# 		indices += [i]

# 	counts = getGaussCounts(len(indices),precision,variance)
# 	normed = normSumPositiveArray(counts)

# 	return normed

# def isValidDist(dist):
# 	if(len(dist.keys()) < 1):
# 		return False
# 	sumProbs = 0
# 	for key in dist:
# 		sumProbs += dist[key]
# 	if(abs(sumProbs - 1.0) > 0.0001):
# 		return False
# 	else:
# 		return True

# def sampleDist(dist):
# 	if(not isValidDist(dist)):
# 		print "sampled invalid dist: " + str(dist)
# 		assert(False)

# 	boundaries = []
# 	output = []
# 	runningSum = 0
# 	for i in xrange(len(dist.keys())):
# 		key = dist.keys()[i]
# 		val = dist[key]
# 		runningSum += val
# 		boundaries += [runningSum]
# 		output += [key]

# 	sample_val = random.random()
# 	for i in xrange(len(boundaries)):
# 		if(sample_val <= boundaries[i]):
# 			return output[i]
# 	return output[-1]

# gauss = getGaussDist(points=256,precision=100000,variance=40)

# g.drawRectangle(-W/2,-H/2,W/2,H/2,fill='white')
# g.drawLine(-W/2,0,W/2,0,fill='black')
# for i in xrange(len(gauss)):
# 	x = -250 + (1*i)
# 	h = -4000 * gauss[i]
# 	g.drawRectangle(x,h,x+1,0,fill='black')
# g.waitForClick()

incrementing = []
for i in xrange(256):
	incrementing += [i]

random.shuffle(incrementing)
random.shuffle(incrementing)
random.shuffle(incrementing)

print incrementing

gauss1 = []
for i in xrange(256):
	gauss1 += [round(random.gauss(0.0,1.0),6)]

#print gauss1
