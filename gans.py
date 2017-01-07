import mxnet as mx
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from scipy.stats import gaussian_kde

#create the graph figure

def make_gans(num_hid):
    gnet = mx.sym.Variable('rand')
    gnet = mx.sym.FullyConnected(data=gnet, name='gc1', num_hidden=num_hid)
    gnet = mx.sym.LeakyReLU(data=gnet, name="leakyrelu1", act_type='leaky')
    gnet = mx.sym.FullyConnected(data=gnet, name='gc2', num_hidden=num_hid)
    gnet = mx.sym.LeakyReLU(data=gnet, name="leakyrelu2", act_type='leaky')
    gnet = mx.sym.FullyConnected(data=gnet, name='gc3', num_hidden=1)  #Is it ok?
    #print gnet.list_outputs()
    #print gnet.list_arguments()

    dnet = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    dnet = mx.sym.FullyConnected(data=dnet, name='dc1', num_hidden=num_hid)
    dnet = mx.sym.Activation(data=dnet, name='relu1', act_type="relu")
    dnet = mx.sym.FullyConnected(data=dnet, name='dc2', num_hidden=num_hid)
    dnet = mx.sym.Activation(data=dnet, name='tanh1', act_type="tanh")
    dnet = mx.sym.FullyConnected(data=dnet, name='dc3', num_hidden=1)
    dnet = mx.sym.Flatten(dnet)  # is it necessarry?
    dnet = mx.sym.LogisticRegressionOutput(data=dnet, label=label, name='loss')

    return gnet, dnet

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.uniform(-1.0, 1.0, shape=(self.batch_size, self.ndim))]

class RealIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('data', (batch_size, ndim))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim))]

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def fentropy(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()


def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))

def vis(outG, outD):
    #numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]
    #Return evenly spaced numbers over a specified interval.
    zs = np.linspace(-1, 1, 128).astype('float32')
    xs = np.linspace(-5, 5, 128).astype('float32')
    ps = gaussian_likelihood(xs, 0.)
    #One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    #A list of one value vector
    gs = outG[0].asnumpy().flatten()
    kde = gaussian_kde(gs)
    score = outD[0].asnumpy().flatten()

    plt.clf()
    plt.plot(xs, ps, '--', lw=2)  #P(Data)
    plt.plot(xs, kde(xs), lw=2)   #G(z)
    plt.plot(xs, score, lw=2)     #D(x)
    plt.xlim([-5., 5.])
    plt.ylim([0., 1.])
    plt.ylabel('Prob')
    plt.xlabel('x')
    plt.legend(['P(data)', 'G(z)', 'D(x)'])
    plt.title('GAN learning guassian')
    fig.canvas.draw()
    plt.show(block=False)
    show()
    plt.pause(0.001)

if __name__ == '__main__':
    fig = plt.figure()
    plt.ion() #It helps the plt moving

    num_hid = 2048
    lr = 0.001
    beta1 = 0.5
    batch_size = 128
    Z = 1
    ctx = mx.cpu()

    rand_iter = RandIter(batch_size, Z)
    real_iter = RealIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    gnet, dnet = make_gans(num_hid)

    # =============module G=============
    modG = mx.mod.Module(symbol=gnet, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
            'decay_factor':0.9999,
        })
    mods = [modG]

    # =============module D=============
    modD = mx.mod.Module(symbol=dnet, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=real_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
            'decay_factor':0.9999,
        })
    mods.append(modD)

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print 'Training...'
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    for epoch in range(1000):
        batch  = real_iter.next()
        rbatch = rand_iter.next()

        #Generate fake data: outG
        modG.forward(rbatch, is_train=True)
        outG = modG.get_outputs()

        label[:] = 0
        modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        outD = modD.get_outputs()
        modD.backward()

        #modD.update()
        gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

        modD.update_metric(mD, [label])
        modD.update_metric(mACC, [label])

        # update discriminator on real
        label[:] = 1
        batch.label = [label]
        modD.forward(batch, is_train=True)
        modD.backward()
        for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        modD.update()

        modD.update_metric(mD, [label])
        modD.update_metric(mACC, [label])

        # update generator
        if epoch % 10 == 0:
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()

        mG.update([label], modD.get_outputs())

        if epoch % 10 == 9:
            vis(outG, outD)
            print 'epoch:', epoch, 'metric:', mACC.get(), 'G:', mG.get(), 'D:', mD.get()
            mACC.reset()
            mG.reset()
            mD.reset()
