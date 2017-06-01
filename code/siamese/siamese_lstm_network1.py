#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#http://www.erogol.com/duplicate-question-detection-deep-learning/
#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from siamese_lstm_network import SiameseNN


class SiameseNNWithDenseLayer(SiameseNN):

    start = 0
    model_file = ".\\siamese.model"
    
    def __init__(self, nfeatures, max_length, vocab_size, embedding_matrix):
        super(SiameseNNWithDenseLayer,self).__init__(nfeatures, max_length, vocab_size, embedding_matrix)
        self.max_length = max_length
        self.n_hidden = nfeatures/2
        self.nfeatures = nfeatures
        self.n_classes = 2
        self.x1 = tf.placeholder(tf.int32, [None,self.max_length]) # batch_size x sentence_length
        self.x2 = tf.placeholder(tf.int32, [None,self.max_length])
        self.y = tf.placeholder(tf.float64, [None,2])
        self.learning_rate = 0.05
		
    def reshape(self,input1,input2,labels=None):
        if input1 != None:
            input1 = np.reshape(input1, (-1, self.max_length))

        if input2 != None:
            input2 = np.reshape(input2, (-1, self.max_length))
        
        if labels != None:
            labels = np.reshape(labels, (-1, 2))
        
        return input1,input2,labels

    def convertLabelsToOneHotVectors(self,labels):
        
        one_hot_label = []
        for label in labels: 
            if label == 0:
              one_hot_label.append([1,0])
            else:
              one_hot_label.append([0,1])
        return one_hot_label
       
    def evaluateResults(self,predictions,actual):
        print(predictions)
        predictions = predictions[0]
        predicted = tf.equal(tf.argmax(predictions, 1), tf.argmax(actual, 1))
        batch_accuracy = tf.reduce_mean(tf.cast(predicted, "float"), name="accuracy")
        batch_accuracy = batch_accuracy.eval()
        
        return batch_accuracy

    def buildSiameseNN(self, left_nn, right_nn):
        #construct fully connected layer
        print(self.nfeatures)
        weights = {
          'out': tf.Variable(tf.random_normal([2*self.nfeatures, self.n_classes],dtype=tf.float64),dtype = tf.float64)
        }
        biases = {
          'out': tf.Variable(tf.random_normal([self.n_classes],dtype=tf.float64),dtype = tf.float64)
        }
           
        joint_layer = tf.concat([left_nn,right_nn],1)
        print("joint layer-->"+str(joint_layer))
        result = tf.nn.softmax(tf.matmul(joint_layer, weights['out']) + biases['out'])
        #add softmax layer
        return result
	
    def optimizeWeights(self,pred):
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return optimizer,cost
     
    def prepareFeatures(self):
        x1 = tf.placeholder(tf.int32, [None,self.max_length]) # batch_size x sentence_length
        x2 = tf.placeholder(tf.int32, [None,self.max_length])
        y = tf.placeholder(tf.float64, [None,2])
        
        return x1,x2,y
		
    def generatePrediction(self,predictions):
        print(predictions[0])
        predicted = tf.argmax(predictions[0],1)
        predicted = predicted.eval()
        return predicted