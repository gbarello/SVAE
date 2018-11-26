import tensorflow as tf

MAX_VAR = 10.

MIN_VAR = 0

def get_var_mat(trans,cor):
#    expt = tf.expand_dims(trans,1)

#    return tf.reduce_sum(tf.transpose(expt,[0,1,3,2])*expt,axis = 2)

    expt = tf.expand_dims(trans,-1)
    exptc = tf.expand_dims(cor,-1)

    diag =  tf.reduce_sum(expt*tf.transpose(expt,[0,3,2,1]),axis = 2)
    offd =  tf.reduce_sum(exptc*tf.transpose(exptc,[0,3,2,1]),axis = 2)

    return diag + offd,diag, offd
