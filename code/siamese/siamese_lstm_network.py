#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#http://www.erogol.com/duplicate-question-detection-deep-learning/
#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class SiameseNN:

    start = 0
    model_file = ".\\siamese.model"
    
    def __init__(self, nfeatures, vocab_size):
        self.nfeatures = nfeatures
        self.n_hidden = 300
        self.n_steps = 300
        self.batch_size = 1000
        self.vocab_size = vocab_size
        
       
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
            input1 = np.reshape(input1, (-1, self.nfeatures,1))

        if input2 != None:
            input2 = np.reshape(input2, (-1, self.nfeatures,1))
        
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
        #distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(left_nn),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(right_nn),1,keep_dims=True))))
        #distance = tf.reshape(distance, [-1], name="distance")
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
        self.x1 = tf.placeholder(tf.float32, [None,self.nfeatures,1])
        self.x2 = tf.placeholder(tf.float32, [None,self.nfeatures,1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        #self.W = tf.Variable(tf.random_normal([self.vocab_size-1,self.nfeatures]),name='embeddings')
        #self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.x1,name="lookup1")
        #self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.x2,name="lookup2")

        
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
            #left_rnn = self.buildRNN(self.embedded_chars1)
        with tf.variable_scope('nn2'):
            right_rnn = self.buildRNN(self.x2)
            #right_rnn = self.buildRNN(self.embedded_chars2)

        self.distance = self.buildSiameseNN(left_rnn,right_rnn)
        
        # Minimize error using cross entropy
        self.pred = tf.reshape(self.distance, [-1], name="distance")
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        

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
                        '''
                        onezero = predict.ravel() < 0.5
                        onezero = [ 1 if i else 0 for i in onezero]
                        predictions = tf.equal(onezero,batch_ys)
                        batch_accuracy = tf.reduce_mean(tf.cast(predictions, "float"), name="accuracy")
                        print("Batch Acc-->"+str(batch_accuracy.eval()))
                        '''
                        # Compute average loss
                        avg_cost += c / total_batch
                        count = count + self.batch_size
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            tf.add_to_collection("distance", self.distance)
            save_path = saver.save(sess, self.model_file)

        print("Optimization Finished!")


    def validateModel(self,test_input1,test_input2,test_labels):
        #test_labels = self.__convertLabelsToOneHotVectors(test_labels)

        test_input1 = np.asarray(test_input1)
        test_input2 = np.asarray(test_input2)
        test_labels = np.asarray(test_labels)

        # tf Graph input
        #self.x1 = tf.placeholder("float", [None,self.nfeatures,self.batch_size],"x1")
        #self.x2 = tf.placeholder("float", [None,self.nfeatures,self.batch_size],"x2")
        #self.y = tf.placeholder("float", [None, 1],"y")

        
        print(len(test_input1))
        print(len(test_input2))
        record_size = len(test_input1)
        
        init = tf.global_variables_initializer()
            
        with tf.Session() as sess:
            sess.run(init)
            #saver = tf.train.import_meta_graph(".\\siamese.model.meta")
            #saver.restore(sess, ".\\siamese.model")
            #self.pred = tf.get_collection("pred")
            print(self.pred)
            overall_accuracy = 0
            
            
            total_batch = int(record_size / self.batch_size)
            for i in range(total_batch):
                batch_x1,batch_x2,batch_ys = self.__createBatch(test_input1,test_input2,test_labels,self.batch_size)
                predictions = self.distance.eval(feed_dict={self.x1: batch_x1, self.x2: batch_x2,self.y: batch_ys})
                # Test model
                predictions = tf.reshape(predictions, [-1], name="distance")
                #Compute Accuracy
                onezero = predictions.ravel() < 0.5
                onezero = [ 1 if i else 0 for i in onezero]
                predictions = tf.equal(onezero,batch_ys)
                batch_accuracy = tf.reduce_mean(tf.cast(predictions, "float"), name="accuracy")
                batch_accuracy = batch_accuracy.eval()
                #batch_accuracy = accuracy.eval({self.x1: batch_x1, self.x2: batch_x2,self.y: batch_ys})				
                overall_accuracy = overall_accuracy + batch_accuracy
                print("Accuracy:", batch_accuracy)
            overall_accuracy = overall_accuracy / total_batch
            print("Overall Accuracy-->"+str(overall_accuracy))
            

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
            saver = tf.train.import_meta_graph(".\\siamese.model.meta")
            saver.restore(sess, ".\\siamese.model.data-00000-of-00001")
            sess.run(init)
            predict = tf.argmax(self.pred,1)
            result = predict.eval(feed_dict={self.x1:test_input1, self.x2:test_input2},session=sess)
            #print(result)
               
        return result