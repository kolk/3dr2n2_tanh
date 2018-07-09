"""
Read the data, divide into train, val, and test sets
For 2 LSTMs - working
Written by Abbhinav Venkat
"""

import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import theano as thea
import numpy as np
from config import cfg
from PIL import Image
import random
import binvox_rw as bvox
from multiprocessing import Queue, Process, Event
import itertools
import time

def findPath():

	train_prop = cfg.train_prop
	no_seq = cfg.no_seq 		# No. of sequences to train on

	# Images
	path = cfg.ip_path
	ip_seq = os.listdir(path)
	ip_seq = ip_seq[0:no_seq]	

	# Ground Truth
	op_path = cfg.op_path
	op_seq = os.listdir(path)
	op_seq = op_seq[0:no_seq]

	tr_y = []
	val_y = []
	test_y = []

	tr_x = []
	val_x = []
	test_x = []

	for folder in ip_seq:
		new_path = path + folder + '/'
		ip_files = os.listdir(new_path)
		ip_files = sorted([new_path + x for x in ip_files])

		new_path = op_path + folder + '/'
		op_files = os.listdir(new_path)
		op_files = sorted([new_path + x for x in op_files])

		tr_size = int(train_prop*len(ip_files))
		val_size = int((len(ip_files) - tr_size)*train_prop)
		test_size = int(len(ip_files) - (tr_size + val_size))
		
		# Ensures that all the data is correctly divided
		assert(tr_size + test_size + val_size == len(ip_files))
		#print((tr_size, val_size, test_size))

		tr_x.extend(ip_files[0:tr_size])
		val_x.extend(ip_files[tr_size:tr_size+val_size])
		#test_x.extend(ip_files[tr_size+val_size:len(ip_files)])

		tr_y.extend(op_files[0:tr_size])
		val_y.extend(op_files[tr_size:tr_size+val_size])
		#test_y.extend(op_files[tr_size+val_size:len(op_files)])
	
	return (tr_x, tr_y, val_x, val_y, test_x, test_y)

def loadData(tr_x, tr_y, train_queue):
	tot_views = cfg.tot_views
	n_views = cfg.n_views
	batch_size = cfg.batch
	n_batch = int(len(tr_x)/batch_size)
	time_len = cfg.time_len

	# Loading one epoch's data 
	global end
	end = 0
	idx = [0]*time_len

	perm_data = []
	#Only sequences of size 3
	for count in range(1, time_len+1):
		temp = (list(itertools.combinations(np.arange(len(tr_x)), count)))
		final = []
		for j in temp:
			flag = 1
			for k in range(len(j)-1):
				diff = j[k+1] - j[k]
				if(diff>3):
					flag = 0
					break
			if(flag==1):
				a = []
				a.extend(j)
				a = list(itertools.permutations(a,len(j)))
				final.extend(a)
		perm_data.append(final)

	while(end!=1):
		n_views = cfg.n_views#np.random.randint(cfg.n_views) + 1        #Randomizing the total no. of views
		#cur_time = np.random.randint(cfg.time_len) + 1
		cur_time = cfg.time_len
		ip = np.zeros((cur_time, n_views, batch_size, 3, cfg.img_h, cfg.img_w), dtype=thea.config.floatX)
		op = np.zeros((batch_size, cfg.n_vox, 2, cfg.n_vox, cfg.n_vox), dtype=thea.config.floatX)

		for bs in range(batch_size): 
			for t in range(cur_time):
				cur_dir = tr_x[perm_data[cur_time-1][idx[cur_time-1]][t]]
				v = random.sample(range(tot_views), n_views) 	#Choosing n_views randomly
				count = 0

				for j in v:
					path = cur_dir + '/' + str(j) + '.png'
					
					#Pre-process the image
					img = Image.open(path)
					img = np.array(img).astype(np.float32)
					#if np.random.rand() > 0.5:
					#	img = img[:, ::-1, ...]

					img = img/255.
					img = img.transpose((2, 0, 1)).astype(thea.config.floatX)
	
					# 3 channels
					ip[t, count, bs, :, :, :] = img
					count = count + 1

			# Ground Truth VoxMesh - Last frame
			cur_path = tr_y[perm_data[cur_time-1][idx[cur_time-1]][cur_time-1]]
			#print(cur_path)
			with open(cur_path, 'rb') as f:
				m1 = bvox.read_as_3d_array(f)
				m1_data = m1.data
				op[bs, :, 0, :, :] = m1_data < 1
				op[bs, :, 1, :, :] = m1_data	     
			
			train_queue.put((ip, op), block=True)

			idx[cur_time-1] = idx[cur_time-1] + 1
			(one, two) = permute(len(perm_data[cur_time-1]), idx[cur_time-1], perm_data[cur_time-1]) 		
			idx[cur_time-1] = one
			perm_data[cur_time-1] = two

def permute(l, idx, perm_data):
	if(idx>=l):
		idx = 0		
		perm_data = np.random.permutation(perm_data)

	return (idx, perm_data)
	
def multiproc(tr_x, tr_y, train_queue):
	
	p = Process(target=loadData, args=(tr_x, tr_y, train_queue))	
	p.start()	
	return p

def kill_processes(queue, processes):

    print('Empty queue')
    while not queue.empty():
        time.sleep(0.5)
        queue.get(False)

    print('kill processes')
    for p in processes:
        p.terminate()

"""
if __name__ == '__main__':
	(tr_x, tr_y, val_x, val_y, test_x, test_y) = findPath()
	(tr_x, tr_y) = loadData(tr_x, tr_y)
	(val_x, val_y) = loadData(val_x, val_y)
"""
