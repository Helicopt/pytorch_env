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

config = utils.config
data_root = config['data']
img_root = data_root + 'images/'

class DataSet(data.Dataset):

	def __init__(self, csvFile, train = True):
		super(DataSet, self).__init__()
		mcsv = csv.DictReader(open(csvFile))
		fields = sorted(mcsv.fieldnames)
		self.data = []
		for it in mcsv:
			la = None
			if 'species' in it:
				la = utils.name2clsid(it['species'])
			sid = int(it['id'])
			im = cv2.imread(img_root+'%d.jpg'%sid)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = im.astype('float32')
			im = cv2.resize(im, (224,224))
			if train:
				ims = utils.dataAuc(im)
			else:
				ims = [im]
			imgs = []
			for im in ims:
				im = im.reshape(1, 224,224)
				imgs.append(im)
			u = []
			for i in fields:
				if i=='species' or i=='id': continue
				u.append(it[i])
			u = np.array(u, dtype='float32')
			sid = np.array([sid], dtype='int')
			if la is None:
				for im in imgs:
					self.data.append((sid, im, u))
			else:
				for im in imgs:
					self.data.append((sid, im, u, np.array([la], dtype='int')))
		self.n = len(self.data)
		print(csvFile, self.n, 'samples')

	def shuffle(self):
		random.shuffle(self.data)

	def __len__(self):
		return self.n
	
	def __getitem__(self, ind):
		return self.data[ind]
