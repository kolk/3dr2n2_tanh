"""
Config setting for the network
"""

from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.tot_views = 12
__C.n_views = 5 # Randomly chosen views

__C.img_w = 255
__C.img_h = 255
__C.n_vox = 128

__C.train_prop = 0.8
__C.no_seq = 2 				# No. of sequences to train on

__C.ip_path = './data/input_depthmaps/'
__C.op_path = './data/groundtruth_meshes/'

__C.batch = 2 
__C.time_len = 2		# Every 3 time sequences in a batc


##############################################
# Loading weights from R2N2
__C.half_init = './output/weights_r2n2.npy'
__C.INIT_R2N2 = True
###############################################

__C.SUB_CONFIG_FILE = []
__C.PROFILE = False

__C.CONST = edict()
__C.CONST.DEVICE = 'gpu'
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 255
__C.CONST.IMG_H = 255
__C.CONST.N_VOX = 128
__C.CONST.N_VIEWS = 5
__C.CONST.BATCH_SIZE = 2
__C.CONST.NETWORK_CLASS = 'ResidualGRUNet'
__C.CONST.WEIGHTS = ''  # when set, load the weights from the file

#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
__C.DIR.OUT_PATH = './output/'

#
# Training
#
__C.TRAIN = edict()

__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.INITIAL_ITERATION = 0  # when the training resumes, set the iteration number
__C.TRAIN.USE_REAL_IMG = False
__C.TRAIN.DATASET_PORTION = [0, 0.8]

__C.TRAIN.NUM_ITERATION = 60000  # maximum number of training iterations
__C.TRAIN.NUM_RENDERING = 10
__C.TRAIN.NUM_VALIDATION_ITERATIONS = 10
__C.TRAIN.VALIDATION_FREQ = 2000
__C.TRAIN.NAN_CHECK_FREQ = 1000
__C.TRAIN.RANDOM_NUM_VIEWS = True  # feed in random # views if n_views > 1

__C.QUEUE_SIZE = 2 # maximum number of minibatches that can be put in a data queue

# Data augmentation
__C.TRAIN.RANDOM_CROP = True
__C.TRAIN.PAD_X = 10
__C.TRAIN.PAD_Y = 10
__C.TRAIN.FLIP = True

# For no random bg images, add random colors
__C.TRAIN.NO_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.RANDOM_BACKGROUND = False
__C.TRAIN.SIMPLE_BACKGROUND_RATIO = 0.5  # ratio of the simple backgrounded images

# Learning
# For SGD use 0.1, for ADAM, use 0.0001
__C.TRAIN.DEFAULT_LEARNING_RATE = 1e-4
__C.TRAIN.POLICY = 'adam'  # def: sgd, adam
# The EasyDict can't use dict with integers as keys
__C.TRAIN.LEARNING_RATES = {'15000': 1e-5, '50000': 1e-6}
__C.TRAIN.MOMENTUM = 0.90
# weight decay or regularization constant. If not set, the loss can diverge
# after the training almost converged since weight can increase indefinitely
# (for cross entropy loss). Too high regularization will also hinder training.
__C.TRAIN.WEIGHT_DECAY = 0.00005
__C.TRAIN.LOSS_LIMIT = 2  # stop training if the loss exceeds the limit
__C.TRAIN.SAVE_FREQ = 1000  # weights will be overwritten every save_freq
__C.TRAIN.PRINT_FREQ = 10

#
# Testing options
#
__C.TEST = edict()
__C.TEST.EXP_NAME = 'test'
__C.TEST.USE_IMG = False
__C.TEST.MODEL_ID = []
__C.TEST.DATASET_PORTION = [0.8, 1]
__C.TEST.SAMPLE_SIZE = 0
__C.TEST.IMG_PATH = ''
__C.TEST.AZIMUTH = []
__C.TEST.NO_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]

__C.TEST.VISUALIZE = False
__C.TEST.VOXEL_THRESH = [0.4]


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
