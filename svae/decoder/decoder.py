import tensorflow as tf
import numpy as np

def make_decoder(latent,nfeat):
    ish = latent.shape
    weights = tf.Variable(name = "decoder_weights",initial_value = np.float32(np.random.normal(0,1./np.sqrt(int(ish[-1])),[nfeat,int(ish[-1])])))
    return tf.reduce_sum(tf.expand_dims(latent,2) * tf.expand_dims(tf.expand_dims(weights,0),0),-1), weights

    
