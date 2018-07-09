"""
Program to convert data to a pre-defined format
Comment out everything else but that step 
"""

import os

path = './data/input_depthmaps/'
seq = os.listdir(path)

for folder in seq:
	new_path = path + folder + '/'
	files = os.listdir(new_path)

	# Step 2 - Create time - directories
	# Dir should not include any made folders
	"""
	for i in range(int(len(files)/12)):
		n = new_path + str(i).zfill(4)
		if(not os.path.isdir(n)):
			cmd = 'mkdir ' + n
			os.system(cmd) 
			#print(cmd)
	"""

	for img in files:
		name = img.split('_')

		if(len(name) > 1):

			#Step 3
			cmd = 'mv ' + new_path + img + ' ' +  new_path + name[0] + '/' + name[1]
			os.system(cmd)
			
			# Step 1 - To rename
			#cmd = 'mv ' + new_path + img + ' ' + new_path + name[s-2] + '_' + name[s-1]
			#os.system(cmd)
			#print(cmd)
			#break


