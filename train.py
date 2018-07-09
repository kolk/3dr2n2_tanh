"""
Main() for training the network
"""
from read_data import *
from res_gru_net import ResidualGRUNet
from solver import Solver
import theano
from config import cfg
import itertools

train_queue, val_queue, p1, p2 = None, None, None, None

def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            kill_processes(train_queue, [p1])
            kill_processes(val_queue, [p2])
            raise

    return func_wrapper

def seq_num(num, end_indices):
	for i in range(len(end_indices)):
		if(num < end_indices[i]):
			return i

def get_combos(listi, end_indices):
	l = []
	for i in range(cfg.time_len):
		a = list(itertools.combinations(range(listi), i+1))                   
		final_lst = []          
		for tup in a:
			add_tup = True
			prev_elem = tup[0]
			prev_seq = seq_num(tup[0], end_indices)
			for j in range(1,i+1):
				if abs(tup[j] - prev_elem) > 1:
					add_tup = False
					break
				seq = seq_num(tup[j], end_indices)
				prev_elem = tup[j]
				if seq != prev_seq:
					add_tup = False
					break
				else:
					prev_seq = seq 
			if add_tup:
				final_lst.append(tup) 
		l.append(final_lst)
	return l

def create_perm(combos):
	perm_data = []
	time_len = cfg.time_len
	for count in range(1, time_len+1):
		final = []
		for j in combos[count-1]:
			flag = 1
			for k in range(len(j)-1):
				diff = j[k+1] - j[k]
				if(diff>1):
					flag = 0
					break
			if(flag==1):
				a = []
				a.extend(j)
				a = list(itertools.permutations(a,len(j)))
				final.extend(a)
		perm_data.append(final)

	return perm_data

@cleanup_handle
def train():
	# Load Data
		
	(tr_x, tr_y, val_x, val_y, test_x, test_y, end_indices) = findPath()
	global train_queue, val_queue, p1, p2
	train_queue = Queue(cfg.QUEUE_SIZE)
	val_queue = Queue(cfg.QUEUE_SIZE)
	
	print('end_indices %s', (end_indices,))
	combos = get_combos(len(tr_x), end_indices)
	print('combos %d' %(len(combos[1])))
	
	l = create_perm(combos)
	print(len(l[1]))
	#print(l[0])
	#print(l[1])	
	p1 = multiproc(tr_x, tr_y, l, train_queue)
	p2 = multiproc(val_x, val_y, l, val_queue)
	
	net = ResidualGRUNet()	
	solver = Solver(net)
	solver.train(train_queue)
	
	kill_processes(train_queue, [p1])
	kill_processes(val_queue, [p2])
	
