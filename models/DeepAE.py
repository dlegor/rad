import tensorflow as tf
import numpy as np

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))


class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list=[7,64,64,7],transfer_function=tf.nn.relu,learning_rate=0.001):
        """input_dim_list must include the original data dimension"""
        #assert len(input_dim_list) < 2
        #raise ValueError(
        #            "Do you need more one layer!")
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.transfer = transfer_function
        self.learning_rate=0.001

        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = 4*np.sqrt(6. / (self.dim_list[i] + self.dim_list[i+1]))

            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],
                                                             np.negative(init_max_value),init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
        ## Decoders parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))
        ## Placeholder for input
        self.input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## coding graph :
        last_layer = self.input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = self.transfer(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        self.hidden = hidden 
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = self.transfer(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
   
        #self.cost = tf.reduce_mean(tf.square(self.input_x - self.recon))
        self.cost =0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.recon, self.input_x), 2.0))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        sess.run(tf.global_variables_initializer())

    def fit(self, X, sess,iteration=100, batch_size=12, init=False,verbose=False):
        assert X.shape[1] == self.dim_list[0]
        
        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]

        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                e,op=sess.run((self.cost,self.train_step),feed_dict = {self.input_x:X[one_batch]})

            if verbose and i%20==0:
                #e = self.cost.eval(session = sess,feed_dict = {self.input_x: X[one_batch]})
                print(" iteration :", i ,", cost:", e)

    def transform(self, X, sess):
        return self.hidden.eval(session = sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session = sess,feed_dict={self.input_x: X})
