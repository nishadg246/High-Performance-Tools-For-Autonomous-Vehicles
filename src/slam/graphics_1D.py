from graphics import *

class graphics_1D:
	def __init__(self,width,height):
		self.win_width = width
		self.win_height = height
		self.win = GraphWin('', width, height)

	def __del__(self):
		self.win.close()





	def offX(self,x):
		return (x + (self.win_width/2))

	def offY(self,y):
		return (y + (self.win_height/2))



	def getPoint(self,x,y,fill='black'):
		pt = Point(self.offX(x),self.offY(y))
		pt.setOutline(fill)
		return pt

	def getLine(self,x0,y0,x1,y1,width=1,fill='black'):
		line = Line(self.getPoint(x0,y0),self.getPoint(x1,y1))
		line.setOutline(fill)
		line.setWidth(width)
		return line




	def drawPoint(self,x,y,fill='black'):
		pt = self.getPoint(x,y,fill)
		pt.draw(self.win)

	def drawCircle(self,x,y,r,fill=None,outline='black'):
		c = Circle(self.getPoint(x,y),r)
		if(fill != None):
			c.setFill(fill)
		if(outline != None):
			c.setOutline(outline)
		c.draw(self.win)

	def drawRectangleFromPoints(self,p0,p1,fill=None,outline='black'):
		r = Rectangle(p0,p1)
		if(fill != None):
			r.setFill(fill)
		if(outline != None):
			r.setOutline(outline)
		r.draw(self.win)

	def drawRectangle(self,x0,y0,x1,y1,fill=None,outline='black'):
		self.drawRectangleFromPoints(self.getPoint(x0,y0),self.getPoint(x0,y0),fill,outline)

	def drawLine(self,x0,y0,x1,y1,width=1,fill='black'):
		l = self.getLine(x0,y0,x1,y1,width,fill)
		l.draw(self.win)

	def drawText(self,x,y,text,fontSize = 5):
		label = Text(self.getPoint(x,y),text)
		label.setSize(fontSize)
		label.draw(self.win)

	def drawWall(self,x,h=80,w=3,fontSize=20,fill='black'):
		self.drawLine(x, h/2, x, -h/2,w,fill)
		self.drawText(x,(h/2) + 2 + (fontSize/2),str(x),fontSize)

	def drawObstacles(self,obs,h=80,w=3,fontSize=20,fill='black'):
		for val in obs:
			self.drawWall(val,h,w,fontSize,fill)

	def drawAxisMarkers(self):
		markers = []
		for i in xrange(10):
			markers += [i*10]

		for i in xrange(1,10):
			markers += [-i*10]

		for mark in markers:
			self.drawWall(mark,h=2,w=1,fontSize=5)

	def drawBackground(self,obs):
		self.drawRectangleFromPoints(Point(0,0),Point(self.win_width,self.win_height),'white','white')
		self.drawLine(-self.win_width/2,0,self.win_width/2,0)
		self.drawObstacles(obs)
		#self.drawAxisMarkers()

	def drawBackgroundRefresh(self,obs):
		p0 = self.getPoint(obs[0]-20,-self.win_height/2)
		p1 = self.getPoint(obs[1]+20,self.win_height/2)
		self.drawRectangleFromPoints(p0,p1,'white','white')

		p2 = self.getPoint(-self.win_width/2,(self.win_height/2)-20)
		p3 = self.getPoint((-self.win_width/2)+20,self.win_height/2)
		self.drawRectangleFromPoints(p2,p3,'white','white')
		self.drawLine(-self.win_width/2,0,self.win_width/2,0)
		self.drawObstacles(obs)
		#self.drawAxisMarkers()

	def drawRobot(self,pos):
		self.drawCircle(pos[0],0,5,outline='red')

	def waitForClick(self):
		self.win.getMouse()

	