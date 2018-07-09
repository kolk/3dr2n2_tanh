import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")
import os
import shutil
import numpy as np
from subprocess import call

from PIL import Image
from config import cfg, cfg_from_list
from solver import Solver
from voxel import voxel2obj
import time
from res_gru_net import ResidualGRUNet

DEFAULT_WEIGHTS = 'output/weights.npy'

def load_demo_images(path):
    ims = []
    for i in range(5):

        im = Image.open(path + '%d.png' % i)
        """
        temp = np.array(im)
        temp2 = np.array([temp, temp, temp])
        """
        # For gray-scale
        #ims.append([temp2.astype(np.float32)/255.])
        # Transpose separates it into 3 matrices of R,G and B

        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])

    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    #pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'
    tic = time.clock()
    # Use the default network model
    net = ResidualGRUNet(compute_grad=False)
    net.load(DEFAULT_WEIGHTS)
    solver = Solver(net)

    toc = time.clock()

    print('Time to load model: ' + str(toc-tic))

    tic = time.clock()
    path = './test/'
    dirs = sorted(os.listdir(path))
    time_ims = []
    for i in dirs:
        print(i)
        time_ims.append(load_demo_images(path + i + '/'))

    # Run the network
    voxel_prediction, _ = solver.test_output(np.array(time_ims))
    pred_file_name = str(i) + '.obj'

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > 0.4)
    toc = time.clock()

    print('Time for each frame: ' + str(float((toc-tic)/len(dirs))))

if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
