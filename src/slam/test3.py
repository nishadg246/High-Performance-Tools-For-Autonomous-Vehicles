from graphics_1D import *
import random

array = [2039,
1051,
1143,
-485,
126,
78,
707,
440,
-133,
-46,
405,
-784,
798,
-429,
1184,
209,
691,
-555,
1107,
69,
-1825,
1163,
758,
-193,
-980,
-1277,
-1401,
-2102,
-298,
586,
-1336,
1180,
-189,
1010,
1600,
-1429,
671,
-595,
1949,
-854,
-530,
493,
517,
-318,
503,
-222,
596,
-2293,
-721,
1581,
-49,
-1311,
-1592,
-1159,
1057,
2887,
-424,
-136,
1251,
-966,
-67,
1417,
704,
-1067]

W = 800
H = 400

g = graphics_1D(W,H)

g.drawRectangle(-W/2,-H/2,W/2,H/2,fill='white')

bins = {}
for i in xrange(len(array)):
	x = int(10*(float(array[i])/1000.0))
	if(x not in bins):
		bins[x] = 1
	else:
		bins[x] += 1

for key in bins:
	count = bins[key]
	h = -count * 5
	x = key
	g.drawLine(x,h,x,0,fill='black')
g.waitForClick()
