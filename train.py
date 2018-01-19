import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable as Var
import torch.optim as optim
import torch.utils.data as data
from math import *
import time
import random
import os
import sys
import argparse
import json
from net import mNet
from dataset import DataSet
import utils
import csv

parser = argparse.ArgumentParser('leaf classification')
parser.add_argument('-t', '--type', type=str, default='train', help='train or test [train]')
parser.add_argument('-l', '--load', type=str, default='', help='model to load [[empty]]')
parser.add_argument('-p', '--param', type=str, default='', help='params to load [[empty]]')

'''
	config for module
	______
'''
config = json.load(open('config.json'))
data_root = config['data']
train_ = data_root+config['train']
val_ = data_root+config['eval']
test_ = data_root+config['test']
snapshot_dir = config['snapshot']
base_lr = config['base_lr']
lr_decay = config['lr_decay']
decay_step = config['decay_step']
max_epoch = config['max_epoch']
result_file = config['result']
dispStep = 10
use_cpu = torch.cuda.device_count()<=0
gpu_num = max(1, torch.cuda.device_count())

'''
	config ends
	______
'''

global Net
Net = mNet()
if use_cpu==False: Net.cuda()
global eval_set, train_set, test_set

def Vari(x):
	if use_cpu:	return Var(x)
	return Var(x.cuda())

def mark(x, y):
	x = np.exp(x)
	n = x.shape[0]
	acc = 0.
	macc = np.zeros(99)
	for i in range(n):
		k = x[i]
		l = y[i]
		ind = k.argmax()
		if ind == l:
			acc += 1
			macc[l] += 1.
	return acc, macc

def do_eval(w = None, p = {}):
	if w is not None:
		if w.isdigit():
			Net.load(snapshot_dir+'__iter%d.pkl'%int(w))
			print 'eval load model  __iter%d.pkl'%int(w)
		else:
			Net.load(w)
			print 'eval load model %s' % w
		Net.eval()
	global eval_set
	dataLoader = data.DataLoader(dataset = eval_set, shuffle = False, num_workers = 8, batch_size = 16 * gpu_num, pin_memory = not use_cpu)
	tot = 0
	acc = 0.
	clsn = 99
	macc = np.zeros(clsn)
	mtot = np.zeros(clsn)
	for batchid, (_, x, y, z) in enumerate(dataLoader):
		bsz = x.size(0)
		x, y, z = Vari(x), Vari(y), z.numpy()
		out = Net(x, y)
		tot += bsz
		for i in z:
			mtot[i] += 1
		acc_, macc_ = mark(out.data.cpu().numpy().reshape(bsz, clsn), z)
		macc += macc_
		acc += acc_
		mAP = 0.
		for i in range(clsn):
			if mtot[i]>0.5:
				mAP += macc[i]/mtot[i]
	for i in range(clsn):
		print '(%d: %d / %d) '%(i, macc[i], mtot[i]), 
	print ''
	print 'all %d input, %d correct' %(tot, acc)
	print 'accuracy: %.3f %%, mAP: %.3f %%'%(acc*100./tot, mAP*100./clsn)
	return acc/tot, mAP/clsn

def report():
	global result_p
	json.dump(result_p, open(result_file, 'w'))

def load_info():
	global result_p
	result_p = {}
	if os.path.exists(result_file):
		result_p = json.load(open(result_file))
	if 'best_acc' not in result_p:
		result_p['best_acc'] = 0.
		result_p['best_it'] = 'NULL'
		result_p['best_model'] = 'NULL'

def do_train(w = None, p = {}):
	start = 0
	load_info()
	if w is None or w=='':
		pass
	elif w.isdigit():
		start = int(w)
		Net.load(snapshot_dir+'__iter%d.pkl'%start)
		print 'train resume from iter %d' %start
	else:
		Net.load(w)
		print 'pretrain load model %s' % w
	global train_set
	bp_params = filter(lambda x: x.requires_grad, Net.parameters())
	opti = optim.SGD(bp_params, lr = base_lr, momentum = 0.9)
	loss_function = nn.NLLLoss()
	Net.train()
	for epoch in range(start, max_epoch):
		train_set.shuffle()
		avg_loss = 0.
		bcnt = 0
		lr = base_lr * (1. - lr_decay) **(epoch//decay_step)
		for pg in opti.param_groups:
			pg['lr'] = lr
		print '\n','-'*16
		print 'epoch [%d] init done, lr = %.6f'%(epoch,lr)
		dataLoader = data.DataLoader(dataset = train_set, shuffle = False, num_workers = 8, batch_size = 16 * gpu_num, pin_memory = not use_cpu)
		for batchid, (_, x, y, z) in enumerate(dataLoader):
			bsz = x.size(0)
			opti.zero_grad()
			x, y, z = Vari(x), Vari(y), Vari(z.squeeze(1))
			out = Net(x, y)
			loss = loss_function(out, z)
			loss.backward()
			loss_ = loss.data[0]
			out_ = out.data.cpu().numpy()
			avg_loss += loss_
			bcnt += 1
			if batchid % dispStep == 0:
				print 'epoch %d [%d] lr: %.6f, batch loss: %.6f, avg loss: %.6f'%(epoch, batchid, lr, loss_, avg_loss/bcnt)
			if isnan(loss_) or isinf(loss_) or abs(loss_)>1e6:
				print '[%d] strange loss result: '%batchid, loss_
			else:
				opti.step()
		print 'epoch %d evaluating...'%epoch
		Net.save(snapshot_dir+'__iter%d.pkl'%(epoch+1))
		print 'model saved.'
		acc, mAP = do_eval(p = p)
		global result_p
		result_p['final'] = mAP
		if mAP > result_p['best_acc']:
			result_p['best_acc'] = mAP
			result_p['best_it'] = '__iter%d'%(epoch+1)
			mod = 'best_%03d.pkl'%int(mAP*1000)
			result_p['best_model'] = mod
			Net.save(snapshot_dir+mod)
			print 'new best!'
		report()
		print 'best_it: %s best_acc: %.3f'%(result_p['best_it'], result_p['best_acc'])

	

def do_test(w, p = {}):
	if w is None or w=='':
		print 'no model loaded'
		exit(233)
	elif w.isdigit():
		Net.load(snapshot_dir+'__iter%d.pkl'%int(w))
		print 'test load model __iter%d.pkl'%int(w)
	else:
		Net.load(w)
		print 'test load model %s' % w
	global test_set
	Net.eval()
	dataLoader = data.DataLoader(dataset = test_set, shuffle = False, num_workers = 8, batch_size = 16 * gpu_num, pin_memory = not use_cpu)

	a = csv.DictReader(open(data_root+'sample_submission.csv'))
	b = csv.DictWriter(open('test_output.csv', 'w'), a.fieldnames)
	res = []
	for batchid, (sid, x, y) in enumerate(dataLoader):
		bsz = x.size(0)
		sid, x, y = sid.numpy(), Vari(x), Vari(y)
		out = Net(x, y)
		out = out.data.cpu().numpy().reshape(bsz, 99)
		out = np.exp(out)
		for i, ii in enumerate(sid):
			u = {}
			u['id'] = int(ii)
			uind = np.argmax(out[i])
			if random.randint(1,10)==1: print '[%03d]'%ii, utils.clsid2name(uind)
			for j in range(99):
				u[utils.clsid2name(j)] = out[i][j]
			res.append(u)
	b.writeheader()
	b.writerows(res)



if __name__=='__main__':
	global eval_set, train_set, test_set
	if use_cpu:	print 'USING CPU ONLY'
	else: print 'USING %d GPU'%gpu_num
	args = parser.parse_args()
	if args.param!='':
		p = json.load(open(args.param))
	else:
		p = {}
	if args.type=='train':
		train_set = DataSet(train_)
		eval_set = DataSet(val_)
		do_train(args.load, p)
	elif args.type=='eval':
		eval_set = DataSet(val_)
		do_eval(args.load, p)
	else:
		test_set = DataSet(test_)
		do_test(args.load, p)
	
