import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
from  .DeepAE import * 
import sklearn.preprocessing as prep


Transf=prep.MinMaxScaler()

def shrink(epsilon, x):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)

    for i in range(len(x)):
        if x[i] > epsilon:
            output[i] = x[i] - epsilon
        elif x[i] < -epsilon:
            output[i] = x[i] + epsilon
        else:
            output[i] = 0
    return output

   
class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model

    MOFICATION: @Daniel Legorreta
    It was adapted for test with Time Series
    Date Modification: August-2018


    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7,transfer_function=tf.nn.sigmoid,learning_rate=0.001):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.transfer_function=transfer_function
        self.error = error
        self.errors=[]
        self.learning_rate=learning_rate
        self.AE = Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes,
            transfer_function=self.transfer_function,learning_rate=self.learning_rate)

    def fit(self, X, sess, inner_iteration = 80,
            iteration=20, batch_size=12, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        #assert X.shape[1] == self.layers_sizes[0]
        #X =(frequency,Cicles)

        #Transf.fit(X.astype('float'))

        #self.X=Transf.transform(X.astype('float'))
        self.X=X

        ## initialize L, S, mu(shrinkage operator)
        
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        #self.L_Aux=np.zeros(X.shape)
        #self.X=X

        mu = (X.size) / (4.0 * nplin.norm(self.X,1))
        self.shrink=self.lambda_ / mu
        print ("shrink parameter:", self.shrink)
        LS0 = self.L + self.S#7X205

        XFnorm = nplin.norm(self.X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            #print ("X2 shape: ", self.X.shape)

            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = self.X - self.S #7X205
            ## Using L to train the auto-encoder
            self.AE.fit(X = self.L, sess = sess,
                                    iteration = inner_iteration,
                                    batch_size = batch_size,
                                    verbose = verbose)
            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)#205X7
            #print ("L_Aux shape: ", self.L_Aux.shape)
            #print(np.sqrt(mean_squared_error(self.X.T.reshape((-1,1)).ravel(),self.L_Aux.reshape((-1,1)).ravel())))

            #self.L=self.L_Aux.T.copy()
            ## alternating project, now project to S
            self.S = shrink(self.shrink, (self.X - self.L).reshape(X.size,order='C')).reshape(X.shape,order='C')
            #print(np.mean(self.S))

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(self.X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
        
        #self.L_final=Transf.inverse_transform(self.L)
        #self.S_final=Transf.inverse_transform(self.S)

        return self.L , self.S

    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)

    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess = sess)
# if __name__ == "__main__":
# 	x = np.load(r"/home/czhou2/Documents/train_x_small.pkl")
# 	sess = tf.Session()
# 	rae = RDAE(sess = sess, lambda_= 2000, layers_sizes=[784,400])

# 	L, S = rae.fit(x ,sess = sess, learning_rate=0.01, batch_size = 40, inner_iteration = 50, 
# 		    iteration=5, verbose=True)

# 	recon_rae = rae.getRecon(x, sess = sess)

# 	sess.close()
# 	print ("cost errors, not used for now:", rae.errors)
# 	from collections import Counter
# 	print ("number of zero values in S:", Counter(S.reshape(S.size))[0])
# 	print ()
