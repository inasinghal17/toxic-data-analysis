import os
import collections
import numpy as np 
import pandas as pd
import nltk
import nltk.corpus
import sklearn
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.linear_model
import nltk.stem.porter
def makeDictionary(root_dir):
	data=pd.read_csv(root_dir)
	all_words=[]
	porter_stemmer=nltk.stem.porter.PorterStemmer()
	stop_words=set(nltk.corpus.stopwords.words('english'))
	new_list=[]

	for i in range(10000):
		sentence=data["comment_text"][i]
		sentence=sentence.lower()
		words=sentence.split()
		final_sentence=[]
		new_sentence=""
		for word in words:
			if word not in stop_words:
				word_stem=porter_stemmer.stem(word)
				final_sentence.append(word_stem)
		new_sentence=" ".join(final_sentence)
		new_list.append(new_sentence)
	return new_list
x=makeDictionary("D://mchine learning bvp//ina_singhal//toxic_challenge//train.csv")
vect=sklearn.feature_extraction.text.CountVectorizer()
f=vect.fit_transform(x)
data=pd.read_csv("D://mchine learning bvp//ina_singhal//toxic_challenge//train.csv")
y1=list()
y2=list()
y3=list()
y4=list()
y5=list()
y6=list()
for q in range(10000):
	y1.append(data["toxic"][q])
	y2.append(data["severe_toxic"][q])
	y3.append(data["obscene"][q])
	y4.append(data["threat"][q])
	y5.append(data["insult"][q])
	y6.append(data["identity_hate"][q])



reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y1,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for toxic:",reg.score(x_test,y_test))

reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y2,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for severe_toxic:",reg.score(x_test,y_test))

reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y3,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for obscene:",reg.score(x_test,y_test))

reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y4,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for threat:",reg.score(x_test,y_test))


reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y5,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for insult:",reg.score(x_test,y_test))


reg=sklearn.linear_model.LogisticRegression()
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(f,y6,test_size=0.2)
reg.fit(x_train,y_train)
print(x_train.shape)
print("for identity_hate:",reg.score(x_test,y_test))
