"""
Read the data, divide into train, val, and test sets
For 2 LSTMs - working
Written by Abbhinav Venkat
"""

import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
from config import cfg
from PIL import Image
import random
import binvox_rw as bvox
from multiprocessing import Queue, Process, Event
from itertools import permutations
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
	end_indices = []
	prev_end = 0
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
		end_indices.append(tr_size+prev_end)
		prev_end += tr_size
	return (tr_x, tr_y, val_x, val_y, test_x, test_y, end_indices)

def loadData(tr_x, tr_y, perm_data, train_queue):
	tot_views = cfg.tot_views
	n_views = cfg.n_views
	batch_size = cfg.batch
	n_batch = int(len(tr_x)/batch_size)
	time_len = cfg.time_len
	train_data_len = len(tr_x)
	print('##############################################################################')	
	# Loading one epoch's data 
	global end
	end = 0
	idx = [0]*time_len

	while(end!=1):
		n_views = cfg.n_views        #Randomizing the total no. of views
		#cur_time = np.random.randint(cfg.time_len) + 1
		cur_time = cfg.time_len
		ip = np.zeros((n_views, cur_time, batch_size, 3, cfg.img_h, cfg.img_w), dtype=np.float32)
		op = np.zeros((batch_size, cfg.n_vox, 2, cfg.n_vox, cfg.n_vox), dtype=np.float32)
		#print('perm_sata len %d ' %(len(perm_data[cfg.time_len-1])))
		#print('cur time %d' %cur_time)
		#print('batch_size %d' %batch_size)
		for bs in range(batch_size): 
			for t in range(cur_time):
				#print('time %d' %t)
				#print(cur_time)
				#print(len(idx))
				#print(len(perm_data), idx[cur_time-1], t)
				#print(type(perm_data), type(perm_data[0]), type(perm_data[0][0]), perm_data[0][0])
				#print(perm_data)
				#print('cur_time %f idx %f perm_data %f'  %(cur_time, idx[cur_time-1], perm_data[cur_time-1][idx[cur_time-1]][t])) 
				cur_dir = tr_x[perm_data[cur_time-1][idx[cur_time-1]][t]]
				#v = random.sample(range(tot_views), n_views) 	#Choosing n_views randomly
				#print(cur_dir)                             
				count = 0
				v = [0, 2, 5, 8, 11] 
				for j in v:
					path = cur_dir + '/' + str(j) + '.png'
					#print('img path %s' %path)	
					#Pre-process the image
					img = Image.open(path)
					img = np.array(img).astype(np.float32)
					'''if np.random.rand() > 0.5:
					    img = np.fliplr(img)
					'''
					img = img/255.
					img = img.transpose((2, 0, 1)).astype(np.float32)
	
					# 3 channels
					ip[count, t,  bs, :, :, :] = img
					count = count + 1

			# Ground Truth VoxMesh - Last frame
			cur_path = tr_y[perm_data[cur_time-1][idx[cur_time-1]][cur_time-1]]
			print(cur_path)
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
	
def multiproc(tr_x, tr_y, l, train_queue):
	
	p = Process(target=loadData, args=(tr_x, tr_y, l, train_queue))	
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
