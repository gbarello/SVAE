import tensorflow as tf
import numpy as np
import utilities as utils
import sys
from run_sparse_vae import prepare_network as prep

def main(directory):

    params = utils.fetch_file(directory + "model_params")

    netparams = prep(params)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    CP = read_ckpt_file(directory + "saved_params/checkpoint")
    
    for c in CP:
        saver.restore(sess,directory + "saved_params/" + c)

        W = sess.run(netparams["weights"])

        if params["pca_truncation"] == "cut":
            np.savetxt(directory + "{}_weights.csv".format(c),netparams["PCA"].inverse_transform(W.transpose()))
        elif params["pca_truncation"] == "smooth":
            np.savetxt(directory + "{}_weights.csv".format(c),W.transpose())
        
    sess.close()

def read_ckpt_file(f):
    F = open(f,"r")
    CP = []
    for l in F:
        CP.append(l.split(":")[-1].split('"')[1])
    return CP
    
if __name__ == "__main__":
    main(sys.argv[1])
