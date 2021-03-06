import tensorflow as tf
import numpy as np 
import sklearn.preprocessing as prep


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)


class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.tanh, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

     
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        self.cost=(tf.norm(self.x-self.reconstruction))**2
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):  
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))   
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32)) 
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32)) 
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))   
        return all_weights

    def partial_fit(self, X):
 
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):        
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
         
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        