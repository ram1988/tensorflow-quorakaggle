import pandas as pd
import re
import math
# import gensim
import editdistance as levendis
#import tensorflow as tf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
from logistic_regression import LogisticClassifier 


class QuoraFeatureGenerator:
   

    def __init__(self):
        train_file = "./train.csv/train.csv"
        test_file = "./test.csv/test.csv"

        # Load Google's pre-trained Word2Vec model.
        # self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
      
      
    def predictTestData(self):
        #predictions = {}
        batch_records = {}

        with open("predicted.csv","a+") as op_file:
            total_count = len(self.test_df) 
            idx = 1
            for test_row in self.test_df.iterrows():
                test_row = test_row[1]
                id = test_row["test_id"]

                print("Processing and Predicting the row-->" + str(id))
                question1 = str(test_row["question1"])
                question2 = str(test_row["question2"])

                test_feature = self.__getTestDataVectors(question1,question2)     
                
                batch_records[id] = test_feature

                if len(batch_records) == 100000:
                    op = self.model.predict(list(batch_records.values()))

                    ct = 0
                    for id in batch_records: 
                        output = str(id)+","+str(op[ct])
                        op_file.write(output+"\n")
                        ct = ct+1 

                    batch_records = {}

            #Remaining batch
            if len(batch_records)!=0:
                op = self.model.predict(list(batch_records.values()))

                ct = 0
                for id in batch_records: 
                    output = str(id)+","+str(op[ct])
                    op_file.write(output+"\n")
                    ct = ct+1 

                
    def trainModel(self):
        df = pd.read_pickle("./train_features.pkl")
        x_df = pd.concat([df.iloc[:,4:6],df.iloc[:,8]],axis=1)
        y_df = df.iloc[:,9]

        print(x_df)
        print(len(x_df))
        print(len(y_df))
    
        train_no = int(0.8 * len(df))
        #train_no = 100000
        print(train_no)

        train_df = x_df.iloc[0:train_no,:]
        train_labels = y_df.iloc[0:train_no]
        test_df = x_df.iloc[train_no:,:]
        test_labels = y_df.iloc[train_no:]

        self.model = LogisticClassifier(3)
        self.model.trainModel(train_df,train_labels)
        self.model.validateModel(test_df,test_labels)

    def generateTFIDFFeatures(self):
        
        total_records = []
        sigmoid = lambda x: 1 / (1 + math.exp(-x))

        joint_df = self.train_df

        iterator = 0
        for df_row in joint_df.iterrows():
            df_row = df_row[1]

            print("Processing the row-->" + str(iterator))

            question1 = str(df_row["question1"])
            question2 = str(df_row["question2"])

            vectorized_q1, vectorized_q2 = self.__getTFIDFVectors(question1, question2)

            question1_tokens = question1.split()
            question2_tokens = question2.split()

            common_words = 0
            for word in question1_tokens:
                if word in question2_tokens:
                    common_words = common_words + 1
            
            cosine_similarity_score = self.__evaluateCosineSimilarity(vectorized_q1, vectorized_q2)
            cosine_similarity_score = cosine_similarity_score[0][0]  
            
            # wmd_score = self.__evaluateWMD(question1,question2)
            levendis_score = sigmoid(self.__evaluateLevensteinDistance(question1, question2))

            record = (vectorized_q1, vectorized_q2, cosine_similarity_score, levendis_score, len(question1_tokens), len(question2_tokens), common_words,
            df_row["is_duplicate"] if "is_duplicate" in df_row else None)
            
            total_records.append(record)
            iterator = iterator+1

        train_len = len(self.train_df)
        self.train_data = total_records[0:train_len]
        
        train_dataframe = DataFrame(data=self.train_data)
        train_dataframe.to_pickle("./train_features.pkl")

    def __getTestDataVectors(self, question1, question2):

        vectorized_q1, vectorized_q2 = self.__getTFIDFVectors(question1, question2)
        sigmoid = lambda x: 1 / (1 + math.exp(-x))

        cosine_similarity_score = self.__evaluateCosineSimilarity(vectorized_q1, vectorized_q2)
        cosine_similarity_score = cosine_similarity_score[0][0]  

        # wmd_score = self.__evaluateWMD(question1,question2)
        levendis_score = sigmoid(self.__evaluateLevensteinDistance(question1, question2))

        question1_tokens = question1.split()
        question2_tokens = question2.split()

        common_words = 0
        for word in question1_tokens:
            if word in question2_tokens:
                common_words = common_words + 1

        record = (cosine_similarity_score, levendis_score, common_words)

        return record
        

    def __getTFIDFVectors(self, question1, question2):
        
        question1 = question1.lower()
        question2 = question2.lower()

        question1 = question1 if question1 != "nan" else ""
        question2 = question2 if question2 != "nan" else ""

        question1 = re.sub('\W+', ' ', question1)
        question2 = re.sub('\W+', ' ', question2)

        question1_tokens = question1.split()
        question2_tokens = question2.split()

        vocabulary = question1_tokens + question2_tokens
        vocabulary = list(set(vocabulary))
 
        vectorizer = TfidfVectorizer(analyzer='word', vocabulary=vocabulary)
        vectorized_q1 = vectorizer.fit_transform([question1])
        vectorized_q2 = vectorizer.transform([question2])

        return vectorized_q1, vectorized_q2
    
    def __evaluateCosineSimilarity(self, question1_vector, question2_vector):
        cosine_score = cosine_similarity(question1_vector, question2_vector)
        return cosine_score

    # normalize value
    def __evaluateWMD(self, question1, question2):
        wmd = self.word2vec_model.wmdistance(question1.lower().split(), question2.lower().split())
        return wmd

    # normalize value
    def __evaluateLevensteinDistance(self, question1, question2):
        leven_dis = levendis.eval(question1.lower(), question2.lower())
        return leven_dis





if __name__ == '__main__':
    featureGenerator = QuoraFeatureGenerator()
    #print(featureGenerator.generateTFIDFFeatures())
    
    featureGenerator.trainModel()
    featureGenerator.predictTestData()
    '''
    df = pd.read_pickle("./train_features.pkl")
    x_df = df.iloc[:,2:7]
    y_df = df.iloc[:,7]

    print(y_df)
    print(len(x_df))
    print(len(y_df))
    
    train_no = int(0.8 * len(df))
    #train_no = 100000
    print(train_no)

    train_df = x_df.iloc[0:train_no,:]
    train_labels = y_df.iloc[0:train_no]
    test_df = x_df.iloc[train_no:,:]
    test_labels = y_df.iloc[train_no:]

    logistic_classifier = LogisticClassifier()
    logistic_classifier.trainModel(train_df,train_labels)
    logistic_classifier.validateModel(test_df,test_labels)
   
    logistic_classifier.predict(x_df)
    '''
    