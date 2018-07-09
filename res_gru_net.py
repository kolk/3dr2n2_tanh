import numpy as np

# Theano
import theano
import theano.tensor as tensor
import datetime as dt
from config import cfg

from layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, ShapeConv3DLayer, \
    EltwiseMultiplyLayer, get_trainable_params, FCConv1DLayer

tensor6 = tensor.TensorType(theano.config.floatX, (False,) * 6)
tensor5 = tensor.TensorType(theano.config.floatX, (False,) * 5)

class LSTM1D(object):
    def __init__(self, x, input_shape):
        n_convfilter = [16, 32, 64, 64, 64, 64]
        n_fc_filters = [1024]
        n_deconvfilter = [64, 64, 64, 16, 8, 2]

        self.x = x
        # To define weights, define the network structure first
        x_ = InputLayer(input_shape)
        conv1a = ConvLayer(x_, (n_convfilter[0], 7, 7))
        conv1b = ConvLayer(conv1a, (n_convfilter[0], 3, 3))
        pool1 = PoolLayer(conv1b)
 
        print('Conv1a = ConvLayer(x, (%s, 7, 7) => input_shape %s,  output_shape %s)' % (n_convfilter[0] , conv1a._input_shape, conv1a._output_shape))
        print('Conv1b = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[0] , conv1b._input_shape, conv1b._output_shape))
        print('pool1 => input_shape %s,  output_shape %s)' % (pool1._input_shape, pool1._output_shape))

        conv2a = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        conv2b = ConvLayer(conv2a, (n_convfilter[1], 3, 3))
        conv2c = ConvLayer(pool1, (n_convfilter[1], 1, 1))
        pool2 = PoolLayer(conv2c)

        print('Conv2a = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[1] , conv2a._input_shape, conv2a._output_shape))
        print('Conv2b = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[1] , conv2b._input_shape, conv2b._output_shape))
        conv3a = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        conv3b = ConvLayer(conv3a, (n_convfilter[2], 3, 3))
        conv3c = ConvLayer(pool2, (n_convfilter[2], 1, 1))
        pool3 = PoolLayer(conv3b)

        print('Conv3a = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[2] , conv3a._input_shape, conv3a._output_shape))
        print('Conv3b = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[2] , conv3b._input_shape, conv3b._output_shape))
        print('Conv3c = ConvLayer(x, (%s, 1, 1) => input_shape %s,  output_shape %s)' % (n_convfilter[1] , conv3c._input_shape, conv3c._output_shape))
        print('pool3 => input_shape %s,  output_shape %s)' % (pool3._input_shape, pool3._output_shape))

        conv4a = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        conv4b = ConvLayer(conv4a, (n_convfilter[3], 3, 3))
        pool4 = PoolLayer(conv4b)

        conv5a = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        conv5b = ConvLayer(conv5a, (n_convfilter[4], 3, 3))
        conv5c = ConvLayer(pool4, (n_convfilter[4], 1, 1))
        pool5 = PoolLayer(conv5b)

        conv6a = ConvLayer(pool5, (n_convfilter[5], 3, 3))
        conv6b = ConvLayer(conv6a, (n_convfilter[5], 3, 3))
        pool6 = PoolLayer(conv6b)

        print('Conv6a = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[5] , conv6a._input_shape, conv6a._output_shape))
        print('Conv6b = ConvLayer(x, (%s, 3, 3) => input_shape %s,  output_shape %s)' % (n_convfilter[5] , conv6b._input_shape, conv6b._output_shape))
        print('pool6 => input_shape %s,  output_shape %s)' % (pool6._input_shape, pool6._output_shape))

        flat6 = FlattenLayer(pool6)
        print('flat6 => input_shape %s,  output_shape %s)' % (flat6._input_shape, flat6._output_shape))

        fc7 = TensorProductLayer(flat6, n_fc_filters[0])
        print('fc7 => input_shape %s,  output_shape %s)' % (fc7._input_shape, fc7._output_shape))

        # Set the size to be 64x4x4x4
        #s_shape_1d = (cfg.batch, n_deconvfilter[0])
        s_shape_1d = (cfg.batch, n_fc_filters[0] )
        self.prev_s = InputLayer(s_shape_1d)
        #view_features_shape = (cfg.batch, n_fc_filters[0], cfg.CONST.N_VIEWS)

        self.t_x_s_update = FCConv1DLayer(
                self.prev_s,
                fc7, n_fc_filters[0],
                isTrainable=True)

        self.t_x_s_reset = FCConv1DLayer(
                self.prev_s,
                fc7, n_fc_filters[0],
                isTrainable=True)

        self.reset_gate = SigmoidLayer(self.t_x_s_reset)
 
        self.rs = EltwiseMultiplyLayer(self.reset_gate, prev_s)
        self.t_x_rs = FCConv1DLayer(self.rs, fc7, n_fc_filters[0], isTrainable=True)

        
        def recurrence(x_curr, prev_s_tensor, prev_in_gate_tensor):
            # Scan function cannot use compiled function.
            input_ = InputLayer(input_shape, x_curr)
            conv1a_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1a.params)
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (n_convfilter[0], 3, 3), params=conv1b.params)
            rect1_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1_)

            conv2a_ = ConvLayer(pool1_, (n_convfilter[1], 3, 3), params=conv2a.params)
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (n_convfilter[1], 3, 3), params=conv2b.params)
            rect2_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (n_convfilter[1], 1, 1), params=conv2c.params)
            res2_ = AddLayer(conv2c_, rect2_)
            pool2_ = PoolLayer(res2_)

            conv3a_ = ConvLayer(pool2_, (n_convfilter[2], 3, 3), params=conv3a.params)
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (n_convfilter[2], 3, 3), params=conv3b.params)
            rect3_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (n_convfilter[2], 1, 1), params=conv3c.params)
            res3_ = AddLayer(conv3c_, rect3_)
            pool3_ = PoolLayer(res3_)

            conv4a_ = ConvLayer(pool3_, (n_convfilter[3], 3, 3), params=conv4a.params)
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (n_convfilter[3], 3, 3), params=conv4b.params)
            rect4_ = LeakyReLU(conv4b_)
            pool4_ = PoolLayer(rect4_)

            conv5a_ = ConvLayer(pool4_, (n_convfilter[4], 3, 3), params=conv5a.params)
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (n_convfilter[4], 3, 3), params=conv5b.params)
            rect5_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (n_convfilter[4], 1, 1), params=conv5c.params)
            res5_ = AddLayer(conv5c_, rect5_)
            pool5_ = PoolLayer(res5_)

            conv6a_ = ConvLayer(pool5_, (n_convfilter[5], 3, 3), params=conv6a.params)
            rect6a_ = LeakyReLU(conv6a_)
            conv6b_ = ConvLayer(rect6a_, (n_convfilter[5], 3, 3), params=conv6b.params)
            rect6_ = LeakyReLU(conv6b_)
            res6_ = AddLayer(pool5_, rect6_)
            pool6_ = PoolLayer(res6_)

            flat6_ = FlattenLayer(pool6_)
            fc7_ = TensorProductLayer(flat6_, n_fc_filters[0], params=fc7.params)
            rect7_ = LeakyReLU(fc7_)

            prev_s_ = InputLayer(s_shape_1d, prev_s_tensor)
            #print(self.prev_s_._output_shape)

            t_x_s_update_ = FCConv1DLayer(
                prev_s_,
                rect7_, n_fc_filters[0],
                params=self.t_x_s_update.params, isTrainable=True)

            t_x_s_reset_ = FCConv1DLayer(
                prev_s_,
                rect7_, n_fc_filters[0],
                params=self.t_x_s_reset.params, isTrainable=True)

            update_gate_ = SigmoidLayer(t_x_s_update_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_)

            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_)
            t_x_rs_ = FCConv1DLayer(
                rs_, rect7_, n_fc_filters[0], params=self.t_x_rs.params, isTrainable=True)

            tanh_t_x_rs_ = TanhLayer(t_x_rs_)

            gru_out_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))

            return gru_out_.output, update_gate_.output

        time_features, _  = theano.scan(recurrence,
            sequences=[self.x],  # along with images, feed in the index of the current frame
            outputs_info=[tensor.zeros_like(np.zeros(s_shape_1d),
                                dtype=theano.config.floatX),
                      tensor.zeros_like(np.zeros(s_shape_1d),
                                 dtype=theano.config.floatX)])
        time_all = time_features[0]
        time_last = time_all[-1]
        
        self.features = time_last

    def feat(self):
        return self.features
 
class Net(object):

    def __init__(self, random_seed=dt.datetime.now().microsecond, compute_grad=True):
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE
        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX
        self.compute_grad = compute_grad

        # (time, views, self.batch_size, 3, self.img_h, self.img_w),
        # override x and is_x_tensor4 when using multi-view network
        self.x = tensor6()
        self.is_x_tensor4 = False

        # (self.batch_size, self.n_vox, 2, self.n_vox, self.n_vox),
        self.y = tensor5()
        self.params_lst = []

        self.activations = []  # list of all intermediate activations
        self.loss = []  # final loss
        self.output = []  # final output
        self.error = []  # final output error
        self.params = []  # all learnable params
        self.grads = []  # will be filled out automatically
        self.setup()

    def setup(self):
        self.network_definition()
        self.post_processing()

    def network_definition(self):
        """ A child network must define
        self.loss
        self.error
        self.params
        self.output
        self.activations is optional
        """
        raise NotImplementedError("Virtual Function")

    def add_layer(self, layer):
        raise NotImplementedError("TODO: add a layer")

    def post_processing(self):
        if self.compute_grad:
            self.grads = tensor.grad(self.loss, [param.val for param in self.params])

    def load_r2n2(self, filename):
        p = np.load(filename)      # Loading numpy files
        px = 0 
        try:
            for idx, param in enumerate(self.params, 0): 
                if idx <= 31: # CNN
                    param.val.set_value(p[px])
                    px += 1
                    print('CNN Init in progress', idx)
                elif idx >= 32 and idx <= 40: # 1st LSTM
                    print('Skipping time LSTM', idx)
                elif idx >= 41 and idx<=49:
                    param.val.set_value(p[px])
                    px += 1
                    print('Loading View LSTM')
                elif idx >=50 and idx <= 75: 
                    print('after 2nd LSTM')
                    param.val.set_value(p[px])
                    px += 1
                else:
                    continue
        except:
            print('INIT ERR')
        print('Done Pre-loading R2N2 weights.')


    def save(self, filename):
        # params_cpu = {}
        params_cpu = []
        for param in self.params:
            # params_cpu[param.name] = np.array(param.val.get_value())
            params_cpu.append(param.val.get_value())
        np.save(filename, params_cpu)
        print('saving network parameters to ' + filename)

    def load(self, filename, ignore_param=True):
        print('loading network parameters from ' + filename)
        params_cpu_file = np.load(filename)
        if filename.endswith('npz'):
            params_cpu = params_cpu_file[params_cpu_file.keys()[0]]
        else:
            params_cpu = params_cpu_file

        succ_ind = 0
        for param_idx, param in enumerate(self.params):
            try:
                param.val.set_value(params_cpu[succ_ind])
                succ_ind += 1
            except IndexError:
                if ignore_param:
                    print('Ignore mismatch')
                else:
                    raise

class ResidualGRUNet(Net):

    def network_definition(self):

        # (multi_views, time, self.batch_size, 3, self.img_h, self.img_w),
        self.x = tensor6()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h
        n_gru_vox = 4
        # n_vox = self.n_vox

        n_convfilter = [16, 32, 64, 64, 64, 64]
        n_fc_filters = [1024]
        n_deconvfilter = [64, 64, 64, 16, 8, 2]

        # Set the size to be 64x4x4x4
        s_shape = (self.batch_size, n_gru_vox, n_deconvfilter[0], n_gru_vox, n_gru_vox)
        # Dummy 3D grid hidden representations
        prev_s = InputLayer(s_shape)

        input_shape = (self.batch_size, 3, img_w, img_h)

        s_shape_1d = (cfg.batch, n_fc_filters[0], )

        lstm1d_all = []
        def get_viewfeats(x_curr):
            lstm1d_all.append(LSTM1D(x_curr, input_shape))
            params_temp = get_trainable_params()
            self.params_lst.append(len(params_temp))
            '''
            count = 0
            for p in params:
                count += 1
            self.param_count
            print('num of params %d' %count)
            '''
            return lstm1d_all[-1].feat()

        view_features_shape = (self.batch_size, n_fc_filters[0])

        view_features, _  = theano.scan(get_viewfeats,
            sequences=[self.x])
        self.view_features = view_features

        fc7 = InputLayer(view_features_shape)        
        t_x_s_update = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), isTrainable=True)
        t_x_s_reset = FCConv3DLayer(prev_s, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), isTrainable=True)

        rll = time_features[0]
        time_last = time_all[-1]

        reset_gate = SigmoidLayer(t_x_s_reset)

        rs = EltwiseMultiplyLayer(reset_gate, prev_s)
        t_x_rs = FCConv3DLayer(rs, fc7, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), isTrainable=True)

        
        def view_rec_test(x_curr, prev_s_tensor, prev_in_gate_tensor):
            count = 0
            params = get_trainable_params()
            for p in params:
                count += 1
            print('view rec test : num of params %d' %count)

            rect8_ = InputLayer(view_features_shape, x_curr) 
            prev_s_ = InputLayer(s_shape, prev_s_tensor)
              
            t_x_s_update_ = FCConv3DLayer(
                prev_s_,
                rect8_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_update.params, isTrainable=True)
            
            t_x_s_reset_ = FCConv3DLayer(
                prev_s_,
                rect8_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3),
                params=t_x_s_reset.params, isTrainable=True)
            
            update_gate_ = SigmoidLayer(t_x_s_update_)
            comp_update_gate_ = ComplementLayer(update_gate_)
            reset_gate_ = SigmoidLayer(t_x_s_reset_)
            
            rs_ = EltwiseMultiplyLayer(reset_gate_, prev_s_)
            t_x_rs_ = FCConv3DLayer(
                rs_, rect8_, (n_deconvfilter[0], n_deconvfilter[0], 3, 3, 3), params=t_x_rs.params, isTrainable=True)
            
            tanh_t_x_rs_ = TanhLayer(t_x_rs_)
        
            gru_out_ = AddLayer(
                EltwiseMultiplyLayer(update_gate_, prev_s_),
                EltwiseMultiplyLayer(comp_update_gate_, tanh_t_x_rs_))
        
            return gru_out_.output, update_gate_.output
        
        s_update, _ = theano.scan(view_rec_test,
            sequences=[view_features],  # along with images, feed in the index of the current frame
            outputs_info=[tensor.zeros_like(np.zeros(s_shape),
                                        dtype=theano.config.floatX),
                       tensor.zeros_like(np.zeros(s_shape),
                                         dtype=theano.config.floatX)])

        
        update_all = s_update[-1]
        s_all = s_update[0]
        s_last = s_all[-1]
        
        #s_last = np.random.rand(self.batch_size, n_gru_vox, n_deconvfilter[0], n_gru_vox, n_gru_vox)
        self.gru_s = InputLayer(s_shape, s_last)
        
        unpool7 = Unpool3DLayer(self.gru_s)
        self.conv7a = Conv3DLayer(unpool7, (n_deconvfilter[1], 3, 3, 3))
        self.rect7a = LeakyReLU(self.conv7a)
        self.conv7b = Conv3DLayer(self.rect7a, (n_deconvfilter[1], 3, 3, 3))
        self.rect7 = LeakyReLU(self.conv7b)
        self.res7 = AddLayer(unpool7, self.rect7)
        
        print('unpool7 => input_shape %s,  output_shape %s)' % (unpool7._input_shape, unpool7._output_shape))      
        
        unpool8 = Unpool3DLayer(self.res7)
        conv8a = Conv3DLayer(unpool8, (n_deconvfilter[2], 3, 3, 3))
        rect8a = LeakyReLU(conv8a)
        self.conv8b = Conv3DLayer(rect8a, (n_deconvfilter[2], 3, 3, 3))
        self.rect8 = LeakyReLU(self.conv8b)
        self.res8 = AddLayer(unpool8, self.rect8)

        print('unpool8 => input_shape %s,  output_shape %s)' % (unpool8._input_shape, unpool8._output_shape))      

        unpool12 = Unpool3DLayer(self.res8)
        conv12a = Conv3DLayer(unpool12, (n_deconvfilter[2], 3, 3, 3))
        rect12a = LeakyReLU(conv12a)
        self.conv12b = Conv3DLayer(rect12a, (n_deconvfilter[2], 3, 3, 3))
        self.rect12 = LeakyReLU(self.conv12b)
        self.res12 = AddLayer(unpool12, self.rect12)

        print('unpool12 => input_shape %s,  output_shape %s)' % (unpool12._input_shape, unpool12._output_shape))      

        unpool9 = Unpool3DLayer(self.res12)
        self.conv9a = Conv3DLayer(unpool9, (n_deconvfilter[3], 3, 3, 3))
        self.rect9a = LeakyReLU(self.conv9a)
        self.conv9b = Conv3DLayer(self.rect9a, (n_deconvfilter[3], 3, 3, 3))
        self.rect9 = LeakyReLU(self.conv9b)
        self.conv9c = Conv3DLayer(unpool9, (n_deconvfilter[3], 1, 1, 1))
        self.res9 = AddLayer(self.conv9c, self.rect9)

        print('unpool9 => input_shape %s,  output_shape %s)' % (unpool9._input_shape, unpool9._output_shape))      

        unpool10 = Unpool3DLayer(self.res9)
        self.conv10a = Conv3DLayer(unpool10, (n_deconvfilter[4], 3, 3, 3))
        self.rect10a = LeakyReLU(self.conv10a)
        self.conv10b = Conv3DLayer(self.rect10a, (n_deconvfilter[4], 3, 3, 3))
        self.rect10 = LeakyReLU(self.conv10b)
        self.conv10c = Conv3DLayer(self.rect10a, (n_deconvfilter[4], 3, 3, 3))
        self.res10 = AddLayer(self.conv10c, self.rect10)

        print('unpool9 => input_shape %s,  output_shape %s)' % (unpool10._input_shape, unpool10._output_shape))      

        self.conv11 = Conv3DLayer(self.res10, (n_deconvfilter[5], 3, 3, 3))
        #self.conv11 = TanhLayer(conv11)
        print('Conv11 = Conv3DLayer(x, (%s, 3, 3, 3) => input_shape %s,  output_shape %s)' % (n_deconvfilter[5] , self.conv11._input_shape, self.conv11._output_shape))
        
     
        #self.conv11 = np.random.rand(cfg.batch, 128, 2, 128, 128)        
        softmax_loss = SoftmaxWithLoss3D(self.conv11.output)
        self.softloss = softmax_loss
        self.loss = softmax_loss.loss(self.y)
        self.error = softmax_loss.error(self.y)
        self.params = get_trainable_params()
        self.output = softmax_loss.prediction()
        #update_all = [1,2,3]        
        self.activations = [update_all]
        
