# avoid decoding problems
import sys
import os 
import pandas as pd
import numpy as np
import spacy
import pickle
import re

from collections import Counter
from pandas import DataFrame
from nltk import word_tokenize
from siamese_lstm_network import SiameseNN
from siamese_lstm_network1 import SiameseNNWithDenseLayer

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
MAX_LENGTH = 30
UNKNOWN = "<UNK>"
PADDING = "<PAD>"

main_vocab = {UNKNOWN:0,PADDING:1}

def loadQuestionsFromTrainDF():
	df = pd.read_csv("..\\train.csv\\train.csv")


	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: str(x).encode("utf-8"))
	df['question2'] = df['question2'].apply(lambda x: str(x).encode("utf-8"))

	return df["question1"],df["question2"],df["is_duplicate"]

def loadQuestionsFromTestDF():
	df = pd.read_csv("..\\test.csv\\test.csv")


	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: str(x).encode("utf-8"))
	df['question2'] = df['question2'].apply(lambda x: str(x).encode("utf-8"))

	return df["question1"],df["question2"]
	
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
			#print(vec)			
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

def prepareTrainData():	
	
	q1,q2,labels = loadQuestionsFromTrainDF() 
	
	tokenized_train_data = []
	vocabularies = []
	pattern = "[^0-9a-zA-Z\\s\\?\\.,]"
	print(len(q1))
	for i in range(0,len(q1)):
		try:
			token1 = re.sub(pattern," ",q1[i].decode("utf-8"))
			token2 = re.sub(pattern," ",q2[i].decode("utf-8"))
			#print(token1)
			#print(token2)
		except UnicodeDecodeError:
			continue
		token1 = word_tokenize(token1.strip().lower())
		token2 = word_tokenize(token2.strip().lower())
		tokenized_train_data.append([token1,token2])
		vocabularies.extend(token1)
		vocabularies.extend(token2)
		
	
	vocabCounter = Counter(vocabularies).most_common()
	idx = len(main_vocab)
	for i in vocabCounter:
		if len(main_vocab) < MAX_NB_WORDS:
			main_vocab[i[0]] = idx
			idx = idx+1
			
	print(len(main_vocab))
	
	print(main_vocab)
	
	
	for i,train_record in enumerate(tokenized_train_data):
		
		qu1 = train_record[0]
		qu2 = train_record[1]
		
		qu1 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu1]
		qu2 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu2]
		
		tokenized_train_data[i] = [qu1,qu2]
		#print(tokenized_train_data[i])
			
	#print(tokenized_train_data)
	
	embedding_matrix = __prepareEmbeddingMatrix(main_vocab)				
	pickle.dump(embedding_matrix,open("embedding_matrix.pkl","wb"))
	pickle.dump(main_vocab,open("main_vocab.pkl","wb"))
	
	return tokenized_train_data,labels,embedding_matrix
	
def prepareTestData():	
	
	q1,q2 = loadQuestionsFromTestDF() 
	
	tokenized_test_data = []
		
	print(len(q1))
	for i in range(0,len(q1)):
		try:
			token1 = re.sub("[^0-9a-zA-Z\\s]","",q1[i].decode("utf-8"))
			token2 = re.sub("[^0-9a-zA-Z\\s]","",q2[i].decode("utf-8"))
			#print(token1)
			#print(token2)
		except UnicodeDecodeError:
			continue
		token1 = word_tokenize(token1.strip().lower())
		token2 = word_tokenize(token2.strip().lower())
		tokenized_test_data.append([token1,token2])
		
	
	for i,test_record in enumerate(tokenized_test_data):
		
		qu1 = test_record[0]
		qu2 = test_record[1]
		
		qu1 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu1]
		qu2 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu2]
		
		tokenized_test_data[i] = [qu1,qu2]
			
			
	print(tokenized_test_data)
	
	
	return tokenized_test_data
	
def __prepareEmbeddingMatrix(vocabulary):	

	nlp = spacy.load('en')
	embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
	
	for word,idx in vocabulary.items():		
		doc = nlp(word) 		
		vec = doc.vector
		embedding_matrix[idx] = vec
	
			
	return embedding_matrix

	

def runModelWithEmbed():
	from keras.preprocessing import sequence
	
	train_data,labels,embedding_matrix = prepareTrainData()
	
	train_q1 = [ rec[0] for rec in train_data]
	train_q2 = [ rec[1] for rec in train_data]
	
	train_q1 = sequence.pad_sequences(train_q1,maxlen=MAX_LENGTH)
	train_q2 = sequence.pad_sequences(train_q2,maxlen=MAX_LENGTH)
	
	train_no = int(0.8 * len(train_q1))
	#train_no = 10000
	#end = 12000
	train_q1 = np.asarray(train_q1)
	train_q2 = np.asarray(train_q2)
	
	train_question1 = train_q1[0:train_no]
	train_question2 = train_q2[0:train_no]
	train_labels = labels[0:train_no]
	
	validate_question1 = train_q1[train_no:]
	validate_question2 = train_q2[train_no:]
	validate_labels = labels[train_no:]
	
	
	print(np.asarray(train_q1).shape)
	vocab_size = len(main_vocab)
	#siamese_nn = SiameseNN(EMBEDDING_DIM, MAX_LENGTH, vocab_size, embedding_matrix)
	siamese_nn = SiameseNNWithDenseLayer(EMBEDDING_DIM, MAX_LENGTH, vocab_size, embedding_matrix)
	siamese_nn.trainModel(train_question1,train_question2,train_labels,one_hot_encoding=True)
	siamese_nn.validateModel(validate_question1,validate_question2,validate_labels,one_hot_encoding=True)
	
	print("testing..")
	test_data = prepareTestData()
	#test_data = test_data[0:10000]
	test_q1 = [ rec[0] for rec in test_data]
	test_q2 = [ rec[1] for rec in test_data]
	
	test_q1 = sequence.pad_sequences(test_q1,maxlen=MAX_LENGTH)
	test_q2 = sequence.pad_sequences(test_q2,maxlen=MAX_LENGTH)
	
	predictions = siamese_nn.predict(test_q1,test_q2)
	pickle.dump(predictions,open("result.pkl","wb"))
	generateResult()
	

def prepareTrainQuestionSets():	
	q1,q2 = loadQuestionsFromTrainDF()
	avg_q1 = prepareQuestion(q1)
	pickle.dump(avg_q1,open("questions1.pkl","wb"))
	avg_q2 = prepareQuestion(q2)
	pickle.dump(avg_q2,open("questions2.pkl","wb"))
	
def prepareTestQuestionSets():	
	q1,q2 = loadQuestionsFromTestDF()
	avg_q1 = prepareQuestion(q1)
	pickle.dump(avg_q1,open("test_questions1.pkl","wb"))
	avg_q2 = prepareQuestion(q2)
	pickle.dump(avg_q2,open("test_questions2.pkl","wb"))
	
def trainModel():
	
	question1 = pickle.load(open("questions1.pkl","rb"))
	question2 = pickle.load(open("questions2.pkl","rb"))
	q1,q2,label = loadQuestionsFromTrainDF()
	
	total_len = len(question1)
	
	print(len(question1))
	train_no = int(0.8 * len(question1))
	#train_no = 10000
	#end = 320000
	print(train_no)
	
	question1 = np.asarray(question1)
	question2 = np.asarray(question2)
	
	train_question1 = question1[0:train_no,0:n_features]
	train_question2 = question2[0:train_no,0:n_features]
	
	train_labels = label[0:train_no]
	test_question1 = question1[train_no:end,0:n_features]
	test_question2 = question2[train_no:end,0:n_features]
	test_labels = label[train_no:]
	
	
	siamese_nn = SiameseNN(n_features,10000)
	print(len(test_question1))
	siamese_nn.trainModel(train_question1,train_question2,train_labels)
	siamese_nn.validateModel(test_question1,test_question2,test_labels)
	testModel(siamese_nn)
	
def testModel(siamese_nn=None):	
	question1 = pickle.load(open("test_questions1.pkl","rb"))
	question2 = pickle.load(open("test_questions2.pkl","rb"))
	print(len(question1))
	print(len(question2))
	
	#siamese_nn = SiameseNN(100,10000)
	question1 = np.asarray(question1)
	question2 = np.asarray(question2)
	question1 = question1[:,0:n_features]
	question2 = question2[:,0:n_features]
	result = siamese_nn.predict(question1,question2)
	pickle.dump(result,open("result.pkl","wb"))
	
	
def generateResult():
	result = pickle.load(open("result.pkl","rb"))
	print(len(result))
	print(np.asarray(result).shape)
	
	
	with open("predicted.csv","a+") as op_file:
		counter = 0
		for batch in result:
			for rec in batch:
				print(counter)
				output = str(counter)+","+str(rec)
				op_file.write(output+"\n")
				counter = counter + 1
	
runModelWithEmbed()
#trainModel()
#testModel()	
#generateResult()
	