import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import h5py
from PIL import Image as im
#from scipy.ndimage import interpolation as inter
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import time
import scipy.io as sio
import os
from numpy.random import permutation
import matplotlib as mtplot
from keras.utils import plot_model

#
Numtrain = 4250

# loading data

#f = h5py.File(r'/run/media/amir/09133421928/fivek_dataset/im_enhancement-master/_scripts/python/Input_image_hists_rgb.mat')
f = h5py.File(r'E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/Input_image_hists_rgb.mat')
#data = f.items()[0][1].value
data = f.get('Input_image_hists_rgb')
InputHists= np.array(data)
InputHists=InputHists.astype(np.float32)
InputHists = InputHists / InputHists.max()
print(InputHists.shape)

#f = h5py.File(r'/run/media/amir/09133421928/fivek_dataset/im_enhancement-master/_scripts/python/Output_image_hists_rgb.mat')
f = h5py.File('E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/Output_image_hists_rgb.mat')
#data = f.items()[0][1].value
data = f.get('Output_image_hists_rgb')
OutputHists= np.array(data)
OutputHists=OutputHists.astype(np.float32)
OutputHists = OutputHists / OutputHists.max()
print(OutputHists.shape)



#f = h5py.File(r'/run/media/amir/09133421928/fivek_dataset/im_enhancement-master/_scripts/python/label_one_shot.mat')
f = h5py.File(r'E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/label_one_shot.mat')
#data = f.items()[0][1].value
data = f.get('label_one_shot')
LabelOneShot = np.array(data)
LabelOneShot=LabelOneShot.astype(np.float32)
print(LabelOneShot.shape)

# Train and Test Set

X_train_input = InputHists[0:Numtrain,:]
X_test_input = InputHists[Numtrain:,:]

X_train_output = OutputHists[0:Numtrain,:]
X_test_output = OutputHists[Numtrain:,:]

y_train = LabelOneShot[0:Numtrain,:]
y_test = LabelOneShot[Numtrain:,:]


# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 100

# dimension of input (and label)
n_x = X_train_input.shape[1]
n_y = y_train.shape[1]

# nubmer of epochs
n_epoch = 100

##  ENCODER ##

# encoder inputs
X = Input(shape=(768, ))
cond = Input(shape=(n_y, ))

# merge pixel representation and label
inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu')(inputs)
# h_q1 = Dense(1024, activation='relu')(h_q)
# h_q2 = Dense(512, activation='relu')(h_q1)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent spa ce
z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##
def sample_ztest(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(1,n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


##  DECODER  ##

# dense ReLU to sigmoid layers
#decoder_hidden1 = Dense(200, activation='relu')
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(768, activation='sigmoid')

z_cond_new = concatenate([X, z, cond])

h_p = decoder_hidden(z_cond_new)
#h_p2 = decoder_hidden2(h_p1)
# h_q1 = Dense(1024, activation='relu')(h_p)
# h_q2 = Dense(256, activation='relu')(h_q1)
# h_q3 = Dense(512, activation='relu')(h_q2)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model([X, cond], outputs)
encoder = Model([X, cond], mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z+n_y+n_x,))
d_h = decoder_hidden(d_in)
#d_h2 = decoder_hidden2(d_h1)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    # norm = K.abs(3 - K.sum(y_pred))
    return 0.7 * recon + 0.3 * kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))

def norm_loss(y_true, y_pred):
	return K.abs(3 - K.sum(y_pred))

def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])
plot_model(cvae, show_shapes= True, show_layer_names=True, to_file='E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/cvae_plot.png')
cvae_hist = cvae.fit([X_train_input, y_train], X_train_output, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test_input[500:], y_test[500:]], X_test_output[500:]),
							callbacks = [EarlyStopping(patience = 5)])

# Test
ztmp = encoder.predict([X_test_input,y_test])
dtmp = np.concatenate([X_test_input,ztmp,y_test],axis=1)
generated = decoder.predict(dtmp)
#np.savetxt('/run/media/amir/09133421928/fivek_dataset/im_enhancement-master/_scripts/python/Output_test.txt', generated)
np.savetxt('E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/Output_test.txt', generated)
sio.savemat('E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/X_test_input.mat',{'X_test_input': X_test_input})
sio.savemat('E:/thesis_phd/Image_enhancement_MrNazemi/git_repo/im_enhancement/Data/generated.mat',{'generated': generated})
#mtplot.pyplot.plot([1,2,3])
index = np.arange(generated.shape[1])
#mtplot.pyplot.subplot(131)
mtplot.pyplot.bar(index,X_test_input[0])
#mtplot.pyplot.subplot(132)
mtplot.pyplot.bar(index,X_test_output[0])
#mtplot.pyplot.subplot(133)
mtplot.pyplot.bar(index,generated[0])
