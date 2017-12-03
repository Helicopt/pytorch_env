import csv
import random

specy = {}
specy_inv = {}

train_data = './train.csv'

def deal():
	a = csv.DictReader(open(train_data))
	cnt = 0
	for i in a:
		if i['species'] not in specy:
			specy[i['species']] = cnt
			specy_inv[cnt] = i['species']
			cnt += 1

deal()

def name2clsid(x):
	if x in specy: return specy[x]
	else: return -1

def clsid2name(x):
	if x in specy_inv: return specy_inv[x]
	else: return ''

if __name__=='__main__':
	a = csv.DictReader(open(train_data))
	mp = {}
	data = [i for i in a]
	random.shuffle(data)
	train_ = []
	val_ = []
	for i in data:
		clsid = name2clsid(i['species'])
		# print i['species'], clsid
		if clsid not in mp:
			mp[clsid] = {'train':0, 'val':0}
		t = mp[clsid]
		if t['val']*10 == t['train']:
			t['val'] += 1
			val_.append(i)
		else:
			t['train'] += 1
			train_.append(i)
	b = csv.DictWriter(open('train_.csv','w'), a.fieldnames)
	b.writeheader()
	b.writerows(train_)
	b = csv.DictWriter(open('val_.csv','w'), a.fieldnames)
	b.writeheader()
	b.writerows(val_)

