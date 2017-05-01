# avoid decoding problems
import sys
import os 
import pandas as pd
import numpy as np
import spacy
import pickle

from siamese_lstm_network import SiameseNN

def loadQuestionsDF():
	df = pd.read_csv("..\\train.csv\\train.csv")


	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: str(x).encode("utf-8"))
	df['question2'] = df['question2'].apply(lambda x: str(x).encode("utf-8"))

	return df["question1"],df["question2"],df["is_duplicate"]

	
def prepareQuestion(question_list):	
	
	nlp = spacy.load('en')

	vecs1 = []
	ct = 1
	for qu in question_list:
		qu = str(qu)
		print(ct)
		doc = nlp(qu) 
		mean_vec = np.zeros([len(doc), 300])
		for word in doc:
			# word2vec
			vec = word.vector
			# fetch df score
			'''
			try:
				idf = word2tfidf[str(word)]
			except:
				#print word
				idf = 0
			# compute final vec
			'''
			mean_vec += vec
		mean_vec = mean_vec.mean(axis=0)
		vecs1.append(mean_vec)
		ct = ct+1
		
	return list(vecs1)

def prepareQuestionSets():	
	q1,q2 = loadQuestionsDF()
	avg_q1 = prepareQuestion(q1)
	pickle.dump(avg_q1,open("questions1.pkl","wb"))
	avg_q2 = prepareQuestion(q2)
	pickle.dump(avg_q2,open("questions2.pkl","wb"))

def trainModel():
	question1 = pickle.load(open("questions1.pkl","rb"))
	question2 = pickle.load(open("questions2.pkl","rb"))
	q1,q2,label = loadQuestionsDF()
	
	train_no = int(0.8 * len(question1))
	print(train_no)
	
	train_question1 = question1[0:train_no]
	train_question2 = question2[0:train_no]
	train_labels = label[0:train_no]
	test_question1 = question1[train_no:]
	test_labels = label[train_no:]
	test_question2 = question2[train_no:]
	
	siamese_nn = SiameseNN(300)
	siamese_nn.trainModel(train_question1,train_question2,train_labels)
	siamese_nn.validateModel(train_question1,train_question2,train_labels)
trainModel()
	
	