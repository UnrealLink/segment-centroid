from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *
import tensorflow as tf
import numpy as np
import ray

class AtariRAMModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self,  
                 k,
                 statedim=(1024,),
                 actiondim=(8,1),
                 hiddenLayerSize=32):
        
        self.hiddenLayerSize = hiddenLayerSize
        self.K = k
        super(AtariRAMModel, self).__init__(statedim, actiondim, k, [0,0],'all')


    def createPolicyNetwork(self):

        #ram inputs in the gym are scaled to 0,1 so don't forget to scale

        # (N x 128 x 256)
        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0]])
        a = tf.placeholder(tf.float32, shape=[None, self.actiondim[0]])
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        # W_h1 = tf.Variable(tf.random_normal([self.statedim[0], self.hiddenLayerSize]))
        # b_1 = tf.Variable(tf.random_normal([self.hiddenLayerSize]))
        # h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_1)

        # W_h2 = tf.Variable(tf.random_normal([self.hiddenLayerSize, self.hiddenLayerSize]))
        # b_2 = tf.Variable(tf.random_normal([self.hiddenLayerSize]))
        # h2 = tf.nn.relu(tf.matmul(h1, W_h2) + b_2)

        # (32 x a)
        output_w = tf.Variable(tf.random_normal([self.statedim[0], self.actiondim[0]]))
        output_b = tf.Variable(tf.random_normal([self.actiondim[0]]))

        #output
        logit = tf.matmul(x, output_w) + output_b

        y = tf.nn.softmax(logit)

        logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=a), [-1,1])

        wlogprob = tf.multiply(weight, logprob)
            
        return {'state': x, 
                    'action': a, 
                    'weight': weight,
                    'prob': y, 
                    'amax': tf.argmax(y, 1),
                    'lprob': logprob,
                    'wlprob': wlogprob,
                    'discrete': True,
                    'output_w': output_w,
                    'output_b': output_b}


    def createTransitionNetwork(self):
        #ram inputs in the gym are scaled to 0,1 so don't forget to scale

        # (N x 128 x 256)
        x = tf.placeholder(tf.float32, shape=[None, self.statedim[0]])
        a = tf.placeholder(tf.float32, shape=[None, 2])
        weight = tf.placeholder(tf.float32, shape=[None, 1])

        # W_h1 = tf.Variable(tf.random_normal([self.statedim[0], self.hiddenLayerSize]))
        # b_1 = tf.Variable(tf.random_normal([self.hiddenLayerSize]))
        # h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_1)

        # W_h2 = tf.Variable(tf.random_normal([self.hiddenLayerSize, self.hiddenLayerSize]))
        # b_2 = tf.Variable(tf.random_normal([self.hiddenLayerSize]))
        # h2 = tf.nn.relu(tf.matmul(h1, W_h2) + b_2)

        # (32 x 1)
        output_w = tf.Variable(tf.random_normal([self.statedim[0], 2]))
        output_b = tf.Variable(tf.random_normal([2]))

        #output
        logit = tf.matmul(x, output_w) + output_b

        y = tf.nn.softmax(logit)

        logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=a), [-1,1])

        wlogprob = tf.multiply(weight, logprob)
            
        return {'state': x, 
                    'action': a, 
                    'weight': weight,
                    'prob': y, 
                    'amax': tf.argmax(y, 1),
                    'lprob': logprob,
                    'wlprob': wlogprob,
                    'discrete': True,
                    'output_w': output_w,
                    'output_b': output_b}




