import pandas as pd
import tensorflow as tf
import numpy as np

class LogisticClassifier:

    start = 0
    
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures
       
    def __createBatch(self,features,labels,batch_size):
        
        self.end = self.start + batch_size		
        
        batch_x = features[self.start:self.end]
        batch_y = labels[self.start:self.end]
        #batch_size = len(batch_y) if len(batch_y) < batch_size else batch_size
        batch_x, batch_y = self.__reshape(batch_x,batch_y)

        self.start = self.end
       
        if(self.end >= len(features)):	
            self.start = 0

        return batch_x, batch_y

    def __convertLabelsToOneHotVectors(self,labels):
        
        one_hot_label = []
        for label in labels: 
            if label == 0:
              one_hot_label.append([1,0])
            else:
              one_hot_label.append([0,1])
        return one_hot_label
		
		
    def __reshape(self,inputs,labels=None):
        if inputs != None:
            inputs = np.reshape(inputs, (-1, self.nfeatures))
        
        if labels != None:
            labels = np.reshape(labels, (-1, 2))
        
        return inputs,labels
		
    #Try scikit, convert to one hot encoding
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    #http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
    #http://stackoverflow.com/questions/34060332/how-to-get-predicted-class-labels-in-tensorflows-mnist-example
    def trainModel(self, input_features, labels):
        # Parameters
        learning_rate = 0.06
        training_epochs = 100
        batch_size = 1000
        display_step = 1

        record_size = len(input_features)
        labels = self.__convertLabelsToOneHotVectors(labels)

        # tf Graph Input
        self.x = tf.placeholder(tf.float32, [None, self.nfeatures])
        self.y = tf.placeholder(tf.float32, [None, 2], name="classes") #convert to one hot encoding

        # Set model weights
        W = tf.Variable(tf.zeros([self.nfeatures, 2]))
        b = tf.Variable(tf.zeros([2]))

        # Construct model
        self.pred = tf.nn.softmax(tf.matmul(self.x, W) + b)  # Softmax

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            
            count = 0
            #input_features = tf.shape(input_features, name=None, out_type=tf.int32)
            input_features = np.asarray(input_features)
            labels = np.asarray(labels)
            
            # Training cycle
            # Change code accordingly
            for epoch in range(training_epochs):
                print("Epoch--->"+str(epoch))
                avg_cost = 0.
                total_batch = int(record_size / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                        batch_xs,batch_ys = self.__createBatch(input_features,labels,batch_size)
                        
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_xs,
                                                                      self.y: batch_ys})
                        # Compute average loss
                        avg_cost += c / total_batch
                        count = count + batch_size
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")


    def validateModel(self,test_inputs,test_labels):
        test_labels = self.__convertLabelsToOneHotVectors(test_labels)

        test_inputs = np.asarray(test_inputs)
        test_labels = np.asarray(test_labels)
        test_inputs,test_labels = self.__reshape(test_inputs,test_labels)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            print("pred-->"+str(correct_prediction))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            print("Accuracy:", accuracy.eval({self.x: test_inputs, self.y: test_labels}))

    def predict(self,test_inputs):
        # Test model
        result = None
        test_inputs = np.asarray(test_inputs)
        test_inputs = self.__reshape(test_inputs)[0]
        #print(test_inputs)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            predict = tf.argmax(self.pred,1)
            result = predict.eval(feed_dict={self.x: test_inputs},session=sess)
            #print(result)
               
        return result