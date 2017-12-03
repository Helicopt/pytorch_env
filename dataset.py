import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable as Var
import torch.utils.data as data
import numpy as np
import random
import os
import sys
import csv
import utils

img_root = './images/'

class DataSet(data.Dataset):

	def __init__(self, csvFile):
		super(DataSet, self).__init__()
		self.csv = csv.DictReader(open(csvFile))
		self.fields = sorted(self.csv.fieldnames)
		self.data = []
		for it in self.csv:
			la = None
			if 'species' in it:
				la = utils.name2clsid(it['species'])
			sid = int(it['id'])
			im = cv2.imread(img_root+'%d.jpg'%sid)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = im.astype('float32')
			im = cv2.resize(im, (224,224))
			im = im.reshape(1, 224,224)
			u = []
			for i in self.fields:
				if i=='species' or i=='id': continue
				u.append(it[i])
			u = np.array(u, dtype='float32')
			sid = np.array([sid], dtype='int')
			if la is None:
				self.data.append((sid, im, u))
			else:
				self.data.append((sid, im, u, np.array([la], dtype='int')))
		self.n = len(self.data)

	def shuffle(self):
		random.shuffle(self.data)

	def __len__(self):
		return self.n
	
	def __getitem__(self, ind):
		# print self.data[ind][0].dtype, self.data[ind][1].dtype, self.data[ind][2].dtype
		return self.data[ind]
