import numpy as np
import cv2
import os
from PyQt4.QtGui import *
from PyQt4 import QtCore
import sys
from math import *
from qimage2ndarray import *
import random
import utils

data_root = utils.config['data']

def randColor(i):
	# return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
	color_ = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,255,255)]
	return color_[i*5%8]

global lines
lines = {i:(randColor(i), 2) for i in range(256)}

def getQImg(im, sz = (900, 900)):
	if im is None:
		exit(233)
	im = cv2.resize(im, sz)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im = array2qimage(im)
	return QPixmap.fromImage(im)

class VPanel(QWidget):

	def __init__(self):
		super(VPanel, self).__init__()

	def initGUI(self):
		self.resize(1920, 1000)
		self.move(0,0)
		self.setWindowTitle('Visualize tool')

		prevButton = QPushButton('Prev')
		nextButton = QPushButton('Next')
		disButton = QPushButton('/')
		self.prevButton = prevButton
		self.nextButton = nextButton
		self.disButton = disButton

		self.imgLabel = QLabel()
		# self.imgLabel.setPixmap(im)

		self.label = QLabel()

		vbox = QVBoxLayout()
		vbox.addStretch(1)
		vbox.addWidget(disButton)
		vbox.addWidget(prevButton)
		vbox.addWidget(nextButton)
		vbox.addWidget(self.label)


		self.cps = [QLabel() for i in range(3)]
		tbox = QVBoxLayout()
		tbox.addWidget(self.cps[0])
		tbox.addWidget(self.cps[1])
		tbox.addWidget(self.cps[2])

		hbox = QHBoxLayout()
		hbox.addWidget(self.imgLabel)
		hbox.addLayout(tbox)
		hbox.addLayout(vbox)

		self.setLayout(hbox)

		self.prevButton.clicked.connect(self.S_prev)
		self.nextButton.clicked.connect(self.S_next)
		self.disButton.clicked.connect(self.S_fromhead)

		return self

	def show(self):
		super(VPanel, self).show()
		return self

	def draw(self):
		self.disButton.setText('%d/%d'%(self.ind, len(self.data)))
		d = self.data[self.ind]
		im = cv2.imread(data_root + 'images/%s.jpg'%d[0])
		im = getQImg(im)
		self.imgLabel.setPixmap(im)
		self.label.setText('predict [%s], prob: %.4f'%(d[1], float(d[2])))
		for i in range(3):
			im = cv2.imread(data_root + 'images/%s.jpg'%self.mp[d[1]][i])
			im = getQImg(im, (300,300))
			self.cps[i].setPixmap(im)

	def S_prev(self):
		self.ind -= 1
		self.ind = max(self.ind, 0)
		self.draw()

	def S_next(self):
		self.ind += 1
		self.ind = min(self.ind, len(self.data)-1)
		self.draw()

	def S_fromhead(self):
		self.setImgSet()

	def keyPressEvent(self, e):
		# print e.key()
		# print [(i,QtCore.Qt.__dict__[i]) for i in dir(QtCore.Qt) if i[:4]=='Key_']
		if e.key() == QtCore.Qt.Key_A or e.key() == QtCore.Qt.Key_W:
			self.S_prev()
		if e.key() == QtCore.Qt.Key_S or e.key() == QtCore.Qt.Key_D:
			self.S_next()
		if e.key() == QtCore.Qt.Key_PageUp:
			for i in range(25):
				self.S_prev()
		if e.key() == QtCore.Qt.Key_PageDown:
			for i in range(25):
				self.S_next()

	def setImgSet(self):
		import csv
		a = csv.DictReader(open('test_output.csv'))
		data = []
		for i in a:
			mx = -1.
			nm = '--'
			for k in i:
				if k=='id': continue
				if float(i[k])>mx:
					mx = float(i[k])
					nm = k
			data.append((i['id'], nm, mx))
		self.data = data
		b = csv.DictReader(open(data_root + 'train.csv'))
		mp = {}
		for i in b:
			nm = i['species']
			if nm not in mp:
				mp[nm] = []
			mp[nm].append(i['id'])
		self.mp = mp
		self.ind = 0
		self.draw()


if __name__=='__main__':
	app = QApplication(sys.argv)

	content = VPanel().initGUI().show()
	content.setImgSet()

	ret_code = app.exec_()
	exit(ret_code)
