#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#http://www.erogol.com/duplicate-question-detection-deep-learning/

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class SiameseNN:

    start = 0
    
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures
        self.n_hidden = 300
        self.n_steps = 300
        self.batch_size = 1000
        
       
    def __createBatch(self,input1,input2,labels,batch_size):
        
        self.end = self.start + batch_size		
        
        batch_x1 = input1[self.start:self.end]
        batch_x2 = input2[self.start:self.end]
        batch_y = labels[self.start:self.end]
        #batch_size = len(batch_y) if len(batch_y) < batch_size else batch_size
        batch_x1, batch_x2, batch_y = self.__reshape(batch_x1,batch_x2,batch_y)

        self.start = self.end
       
        if(self.end >= len(input1)):	
            self.start = 0

        return batch_x1,batch_x2,batch_y

    def __convertLabelsToOneHotVectors(self,labels):
        
        one_hot_label = []
        for label in labels: 
            if label == 0:
              one_hot_label.append([1,0])
            else:
              one_hot_label.append([0,1])
        return one_hot_label
		
		
    def __reshape(self,input1,input2,labels=None):
        if input1 != None:
            input1 = np.reshape(input1, (-1, self.nfeatures,self.batch_size))

        if input2 != None:
            input2 = np.reshape(input2, (-1, self.nfeatures,self.batch_size))
        
        if labels != None:
            labels = np.reshape(labels, (-1, 1))
        
        return input1,input2,labels


    def buildRNN(self,x):
        x = tf.unstack(x, self.n_steps, 1)

        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        print(outputs[-1])

        return outputs[-1]

		
    def buildSiameseNN(self, left_nn, right_nn):
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(left_nn,right_nn)),1,keep_dims=True))
        distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(left_nn),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(right_nn),1,keep_dims=True))))
        distance = tf.reshape(distance, [-1], name="distance")
        return distance

    #http://stackoverflow.com/questions/36844909/siamese-neural-network-in-tensorflow
    #https://github.com/dhwajraj/deep-siamese-text-similarity/blob/master/siamese_network.py
    def trainModel(self, input1, input2, labels):
        # Parameters
        learning_rate = 0.06
        training_epochs = 1
        
        display_step = 1

        record_size = len(input1)
        #labels = self.__convertLabelsToOneHotVectors(labels)

        # Set model weights
        # tf Graph input
        self.x1 = tf.placeholder("float", [None,self.nfeatures,self.batch_size])
        self.x2 = tf.placeholder("float", [None,self.nfeatures,self.batch_size])
        self.y = tf.placeholder("float", [None, 1])

        # Define weights
        '''
        weights = {
             'out': tf.Variable(tf.random_normal([300, 2]))
        }
        biases = {
             'out': tf.Variable(tf.random_normal([2]))
        }
        '''

        with tf.variable_scope('nn1'):
            left_rnn = self.buildRNN(self.x1)
        with tf.variable_scope('nn2'):
            right_rnn = self.buildRNN(self.x2)

        self.pred = self.buildSiameseNN(left_rnn,right_rnn)
        
        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        

        with tf.Session() as sess:
            sess.run(init)
            
            count = 0
            #input_features = tf.shape(input_features, name=None, out_type=tf.int32)
            input1 = np.asarray(input1)
            input2 = np.asarray(input2)
            labels = np.asarray(labels)
            
            # Training cycle
            # Change code accordingly
            for epoch in range(training_epochs):
                print("Epoch--->"+str(epoch))
                avg_cost = 0.
                total_batch = int(record_size / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                        print("batch--->"+str(i))
                        batch_x1,batch_x2,batch_ys = self.__createBatch(input1,input2,labels,self.batch_size)
                        
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([optimizer, cost], feed_dict={self.x1: batch_x1, self.x2: batch_x2,
                                                                      self.y: batch_ys})
                        # Compute average loss
                        avg_cost += c / total_batch
                        count = count + self.batch_size
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            save_path = self.saver.save(sess, "siamese.model")

        print("Optimization Finished!")


    def validateModel(self,test_input1,test_input2,test_labels):
        #test_labels = self.__convertLabelsToOneHotVectors(test_labels)

        test_input1 = np.asarray(test_input1)
        test_input2 = np.asarray(test_input2)
        test_labels = np.asarray(test_labels)
        test_input1,test_input2,test_labels = self.__reshape(test_input1,test_input2,test_labels)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.saver.restore(sess, "siamese.model")
            sess.run(init)
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            print("pred-->"+str(correct_prediction))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            print("Accuracy:", accuracy.eval({self.x1: batch_x1, self.x2: batch_x2,self.y: batch_ys}))

    def predict(self,test_input1,test_input2):
        # Test model
        result = None
        test_input1 = np.asarray(test_input1)
        test_input2 = np.asarray(test_input2)
        result = self.__reshape(test_input1,test_input2)
        test_input1 = result[0]
        test_input2 = result[1]
        #print(test_inputs)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.saver.restore(sess, "siamese.model")
            sess.run(init)
            predict = tf.argmax(self.pred,1)
            result = predict.eval(feed_dict={self.x1:test_input1, self.x2:test_input2},session=sess)
            #print(result)
               
        return result