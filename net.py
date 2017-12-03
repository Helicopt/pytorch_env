import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable as Var
import numpy as np
import random
import copy
from math import *

class mNet(nn.Module):

	def __init__(self):
		super(mNet, self).__init__()
		self.conva1 = nn.Conv2d(1,16,3, stride = 2, padding = 1) #112
		self.conva2 = nn.Conv2d(16,16,3, stride = 1, padding = 1) #112
		self.bna = nn.BatchNorm2d(16)
		self.convb1 = nn.Conv2d(16,32,3, stride = 2, padding = 1) #28
		self.convb2 = nn.Conv2d(32,32,3, stride = 1, padding = 1) #28
		self.convb3 = nn.Conv2d(32,32,3, stride = 1, padding = 1) #28
		self.bnb = nn.BatchNorm2d(32)
		self.convc1 = nn.Conv2d(32,64,3, stride = 2, padding = 1) #7
		self.convc2 = nn.Conv2d(64,64,3, stride = 1, padding = 1) #7
		self.convc3 = nn.Conv2d(64,64,3, stride = 1, padding = 1) #7
		self.bnc = nn.BatchNorm2d(64)
		self.fconv = nn.Conv2d(64,64,1, stride = 1)

		self.fc0a = nn.Linear(192, 384)
		self.fc0b = nn.Linear(384, 144)

		self.fc1a = nn.Linear(64 * 7 * 7, 512)
		self.fc1b = nn.Linear(512, 256)

		self.fc2 = nn.Linear(400, 256)
		self.fc3 = nn.Linear(256, 99)

		self.sm = nn.LogSoftmax()

	def forward(self, x, y):
		n = x.size(0)
		x = F.max_pool2d(F.relu(self.bna(self.conva2(self.conva1(x)))), 2)
		#print x.size()
		x = F.max_pool2d(F.relu(self.bnb(self.convb3(self.convb2(self.convb1(x))))), 2)
		#print x.size()
		x = F.relu(self.bnc(self.convc3(self.convc2(self.convc1(x)))))
		#print x.size()
		x = self.fconv(x).view(n, -1)
		#print x.size()
		y = F.relu(self.fc0b(F.relu(self.fc0a(y))))
		x = F.relu(self.fc1b(F.relu(self.fc1a(x))))
		#print x.size()
		z = torch.cat([x, y], 1)
		#print z.size()
		z = F.relu(self.fc2(z))
		z = self.fc3(z)
		#print z.size()
		out = self.sm(z)
		return out
	
	def load(self, fn):
		with open(fn, 'rb') as f:
			pre = torch.load(f)
			dic = self.state_dict()
			pre = {k:v for k, v in pre.items() if k in dic}
			dic.update(pre)
			self.load_state_dict(dic)

	def save(self, fn):
		torch.save(self.state_dict(), fn)

if __name__=='__main__':
	net = mNet()
	print net
