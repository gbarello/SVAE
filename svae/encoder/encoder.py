import tensorflow as tf
import numpy as np

corscale = 1

def make_encoder(input_tensor,latent_size,n_extra_dim):
    
    net = tf.layers.dense(input_tensor,128,activation = tf.nn.relu)
    
    net1 = tf.layers.dense(net,256,activation = tf.nn.relu)
    net1 = tf.layers.dense(net1,512,activation = tf.nn.relu)
    
    net2 = tf.layers.dense(net,256,activation = tf.nn.relu)
    net2 = tf.layers.dense(net2,512,activation = tf.nn.relu)        

    mean = tf.layers.dense(net1,latent_size)

    var = tf.layers.dense(net2,latent_size,activation = lambda x: .01 + tf.nn.sigmoid(x))
    var = tf.map_fn(tf.diag,var)

    if n_extra_dim > 0:
        var2 = tf.layers.dense(net2,latent_size*n_extra_dim,activation = lambda x: corscale*tf.nn.tanh(x))
        var2 = tf.reshape(var2,[-1,latent_size,n_extra_dim])
        var2pad = tf.zeros([var2.shape[0],latent_size,latent_size - n_extra_dim])
        var2 = tf.concat([var2,var2pad],axis = 2)
        
    else:
        var2 = tf.zeros(var.shape)
        
    return mean,var,var2
