#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#http://www.erogol.com/duplicate-question-detection-deep-learning/
#https://github.com/jeromeyoon/Tensorflow-siamese/blob/master/main.py
#https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class SiameseNN:

    start = 0
    model_file = ".\\siamese.model"
    
    def __init__(self, nfeatures, max_length, vocab_size, embedding_matrix):
        self.nfeatures = nfeatures
        self.n_hidden = nfeatures/2
        self.n_steps = max_length
        self.n_layers = 5
        self.batch_size = 1000
        self.dropout = 0.75
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.threshold = 0.7
        self.learning_rate = 0.05
        self.epsilon = 1e-3
       
       
    def __createBatch(self,input1=None,input2=None,labels=None,batch_size=None):
        
        self.end = self.start + batch_size		
        
        batch_x1 = input1[self.start:self.end]
        batch_x2 = input2[self.start:self.end]
        print(len(batch_x1))
     
        if labels!=None:
           batch_y = labels[self.start:self.end]
        #batch_size = len(batch_y) if len(batch_y) < batch_size else batch_size
        #batch_x1, batch_x2, batch_y = self.__reshape(batch_x1,batch_x2,batch_y)

        self.start = self.end
       
        if(self.end >= len(input1)):	
            self.start = 0

        if labels!=None:
           print(len(batch_x1))
           #batch_x1,batch_x2,batch_y = self.__reshape(batch_x1,batch_x2,batch_y)
           print(len(batch_x1))
           return batch_x1,batch_x2,batch_y
        else:
           print(batch_x1.shape)
           #batch_x1,batch_x2,labels = self.__reshape(batch_x1,batch_x2)
           print(batch_x1.shape)
           return batch_x1,batch_x2

    def convertLabelsToOneHotVectors(self,labels):
        
        one_hot_label = []
        for label in labels: 
            if label == 0:
              one_hot_label.append([1,0])
            else:
              one_hot_label.append([0,1])
        return one_hot_label
		
		
    def reshape(self,input1,input2,labels=None):
        if input1 != None:
            input1 = np.reshape(input1, (-1, self.max_length))

        if input2 != None:
            input2 = np.reshape(input2, (-1, self.max_length))
        
        if labels != None:
            labels = np.reshape(labels, (-1, 1))
        
        return input1,input2,labels

    def insertBatchNNLayer(self,mat_rel,axes,dimension_size):
        batch_mean, batch_var = tf.nn.moments(mat_rel,axes)
        scale2 = tf.Variable(tf.ones(dimension_size,dtype=tf.float64),dtype=tf.float64)
        beta2 = tf.Variable(tf.zeros(dimension_size,dtype=tf.float64),dtype=tf.float64)
        bn_layer = tf.nn.batch_normalization(mat_rel,batch_mean,batch_var,beta2,scale2,self.epsilon)
         
        return bn_layer

	
    def buildRNN(self,x,scope):
        print(x)
        x = tf.transpose(x, [1, 0, 2])        
        #print(x)
        x = tf.reshape(x, [-1,self.nfeatures])
        #print(x)
        x = tf.split(x, self.n_steps, 0)
        print(x)
        #lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0) for _ in range(self.n_layers)], state_is_tuple=True)
        #outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float64)
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            fw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                #fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout)                
                fw_cell_array.append(fw_cell)
            fw_cell = rnn.MultiRNNCell(fw_cell_array, state_is_tuple=True)
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            bw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                bw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                #bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout)
                bw_cell_array.append(bw_cell)
            bw_cell = rnn.MultiRNNCell(bw_cell_array, state_is_tuple=True)

        outputs, _,_ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float64)
        #outputs, = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float64)

        
        print(outputs)
        print(outputs[-1])
        
        return outputs[-1]

    def optimizeWeights(self,pred):
        # Minimize error using cross entropy
        self.pred = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        return optimizer,cost
		
    def buildSiameseNN(self, left_nn, right_nn):
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(left_nn,right_nn)),1,keep_dims=True))
        distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(left_nn),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(right_nn),1,keep_dims=True))))
        distance = tf.reshape(distance, [-1], name="distance")
        return distance

    def prepareFeatures(self):
        x1 = tf.placeholder(tf.int32, [None,self.max_length]) # batch_size x sentence_length
        x2 = tf.placeholder(tf.int32, [None,self.max_length])
        y = tf.placeholder(tf.float64, [None,1])  
        
        return x1,x2,y
        

    #http://stackoverflow.com/questions/36844909/siamese-neural-network-in-tensorflow
    #https://github.com/dhwajraj/deep-siamese-text-similarity/blob/master/siamese_network.py
    def trainModel(self, input1, input2, labels, one_hot_encoding=False):
        # Parameters
        
        training_epochs = 4
        
        display_step = 1

        record_size = len(input1)
 
        if(one_hot_encoding):
             labels = self.convertLabelsToOneHotVectors(labels)

        # Set model weights
        # tf Graph input
        #self.x1 = tf.placeholder(tf.int32, [None,self.max_length]) # batch_size x sentence_length
        #self.x2 = tf.placeholder(tf.int32, [None,self.max_length])
        #self.y = tf.placeholder(tf.float64, [None,1])
        
        #self.embedded_chars1 = tf.nn.embedding_lookup(self.embedding_matrix, self.x1,name="lookup1") # batch_size x sent_length x embedding_size
        #self.embedded_chars2 = tf.nn.embedding_lookup(self.embedding_matrix, self.x2,name="lookup2")
        
        self.x1,self.x2,self.y = self.prepareFeatures()
        self.embedded_chars1 = tf.nn.embedding_lookup(self.embedding_matrix, self.x1,name="lookup1") # batch_size x sent_length x embedding_size
        self.embedded_chars2 = tf.nn.embedding_lookup(self.embedding_matrix, self.x2,name="lookup2")
       
        print("Embedding-->"+str(self.embedded_chars1))
        print("Embedding-->"+str(self.embedded_chars2))
        print("Embedding-->"+str(self.x1))
        print("Embedding-->"+str(self.x2))
       
        with tf.variable_scope('nn1') as scope1:
            left_rnn = self.buildRNN(self.embedded_chars1,"nn1_side")
        with tf.variable_scope('nn2') as scope2:
            right_rnn = self.buildRNN(self.embedded_chars2,"nn2_side")
           
        self.pred = self.buildSiameseNN(left_rnn,right_rnn)
        
        # Minimize error using cross entropy
        '''
        self.pred = self.distance 
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        '''
        # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        optimizer,cost = self.optimizeWeights(self.pred)
		

        with tf.Session() as sess:
            sess.run(init)
            
            count = 0
            #input_features = tf.shape(input_features, name=None, out_type=tf.int32)
            input1 = np.asarray(input1)
            input2 = np.asarray(input2)
            labels = np.asarray(labels)
 
            input1, input2, labels = self.reshape(input1,input2,labels)
            
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
                    #-1304 cost :0
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            tf.add_to_collection("distance", self.pred)
            tf.add_to_collection("x1", self.x1)
            tf.add_to_collection("x2", self.x2)
            tf.add_to_collection("y", self.y)
            save_path = saver.save(sess, self.model_file)

        print("Optimization Finished!")


    def validateModel(self,test_input1,test_input2,test_labels,one_hot_encoding=False):
        
        if one_hot_encoding:
            test_labels = self.convertLabelsToOneHotVectors(test_labels)

        test_input1 = np.asarray(test_input1)
        test_input2 = np.asarray(test_input2)
        test_labels = np.asarray(test_labels)

        print("Test1--->"+str(len(test_input1)))
        print("Test2--->"+str(len(test_input2)))
        
         
        test_input1,test_input2,test_labels = self.reshape(test_input1,test_input2,test_labels)

        # tf Graph input
        #self.x1 = tf.placeholder("float", [None,self.nfeatures,1],"x1")
        #self.x2 = tf.placeholder("float", [None,self.nfeatures,1],"x2")
        #self.y = tf.placeholder("float", [None, 1],"y")

        
        print(len(test_input1))
        print(len(test_input2))
        print(len(test_labels))
        record_size = len(test_input1)
        
        init = tf.global_variables_initializer()
            
        with tf.Session() as sess:
            sess.run(init)
            #saver = tf.train.import_meta_graph(".\\siamese.model.meta")
            #saver.restore(sess, ".\\siamese.model")
            #self.distance = tf.get_collection("distance")[0]
            #print(self.distance)
            overall_accuracy = 0
            
            
            total_batch = int(record_size / self.batch_size)
            for i in range(total_batch):
                batch_x1,batch_x2,batch_ys = self.__createBatch(test_input1,test_input2,test_labels,self.batch_size)
                print(len(batch_x1))
                predictions = sess.run([self.pred], feed_dict={self.x1: batch_x1, self.x2: batch_x2})
                #Compute Accuracy
                batch_accuracy = self.evaluateResults(predictions,batch_ys)
                overall_accuracy = overall_accuracy + batch_accuracy
                print("Accuracy:", batch_accuracy)
            overall_accuracy = overall_accuracy / total_batch
            print("Overall Accuracy-->"+str(overall_accuracy))
 
    def evaluateResults(self,predictions,actual):
        predictions = np.asarray(predictions)
        #print(predictions) 
        onezero = predictions.ravel() < self.threshold
        onezero = [ 1 if i else 0 for i in onezero]
        print(onezero)
        print(len(onezero))
        predictions = tf.equal(predicted,actual)
        batch_accuracy = tf.reduce_mean(tf.cast(predictions, "float"), name="accuracy")
        batch_accuracy = batch_accuracy.eval()
        
        return batch_accuracy

    def predict(self,test_input1,test_input2):
        # Test model
        result = []
        test_input1 = np.asarray(test_input1)
        test_input2 = np.asarray(test_input2) 
        record_size = len(test_input1)
        
        #test_input1 = result[0]
        #test_input2 = result[1]
        #print(test_inputs)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            '''
            saver = tf.train.import_meta_graph(".\\siamese.model.meta")
            saver.restore(sess, ".\\siamese.model")
            self.distance = tf.get_collection("distance")[0]
            self.x1 = tf.get_collection("x1")[0]
            self.x2 = tf.get_collection("x2")[0]
            '''
            total_batch = int(record_size / self.batch_size)+1

            for i in range(total_batch):            
                batch_x1,batch_x2 = self.__createBatch(test_input1,test_input2,batch_size=self.batch_size)
                
                print(len(batch_x1))
                predictions = sess.run([self.pred], feed_dict={self.x1: batch_x1, self.x2: batch_x2})
                onezero = self.generatePrediction(predictions)
                
                print(onezero)
                print(len(onezero))
                result.append(onezero)
                #print(result)
               
        return result

    def generatePrediction(self,predictions):
        predictions = np.asarray(predictions)[0]
        onezero = predictions.ravel() < self.threshold
        onezero = [ 1 if i else 0 for i in onezero]

        return onezero