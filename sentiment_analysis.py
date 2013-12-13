%matplotlib inline
from __future__ import division
import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

########## FUNCTIONS ##########

###### Extract positive and negative reviews #####
def extract_positive_negative_reviews(df,num=100):
    positive = df[df.stars==5]
    negative = df[df.stars==1]
    positive = positive.iloc[:num]
    negative = negative.iloc[:num]
    positive = positive[['text','stars']]
    negative = negative[['text','stars']]
    positive.stars = positive.stars.replace(5,1)
    negative.stars = negative.stars.replace(1,0)
    train_df = pd.concat([positive,negative])
    return train_df

##### Clean the reviews before Sentiment Analysis #####
def clean(df):
    temp_df = df.copy()
    #REMOVING DIRTY THINGS
    print 'Removing Dirty Stuff...'
    for i,tweet in enumerate(temp_df.text):
        #Remove hyperlinks
        temp = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tweet)
        #Remove quotes
        temp = re.sub(r'&quot;|&amp&#39;', '', temp)
        #Remove citations
        temp = re.sub(r'@[a-zA-Z0-9]*', '', temp)
        #Remove tickers
        temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
        #Remove numbers
        temp = re.sub(r'[0-9]*','',temp)
        temp_df.text.iloc[i] = temp
        
    #REMOVE PUNCTUATION
    print 'Remove Punctuation'
    temp_df.text = [filter(lambda x: x not in string.punctuation,tweet) for tweet in temp_df.text]
        
    return temp_df 

##### Tokenize the words #####
def tokenize(df,max_features = 200):
    print 'Start Tokenizing Using Tf-Idf'
    vectorizer = CountVectorizer(min_df=10,stop_words='english',binary = True, max_features = max_features)
    X = vectorizer.fit_transform(df.text)
    X = X.toarray()
    return X,vectorizer

##### Apply Naive Bayes Model #####
def Naive_Bayes(df,X):
    clf = MultinomialNB(fit_prior=False)
    clf.fit(X, list(df.stars))
    print 'Accuracy of Naive Bayes =%f'% (sum(clf.predict(X)==df.stars)/len(df.stars))

##### Apply Naive Bayes Model with K2 (1 parent) #####
def Naive_Bayes_K2_1_parent(df,X,vectorizer):
    parents = pickle.load(open('parents.txt'))
    features = vectorizer.get_feature_names()
    Counts = pd.DataFrame(X)
    Counts.columns = features
    
    #Dictionnary of (Key, Values) = (Word, Parents)
    d = {}
    for i,ele in enumerate(features):
        d[ele] = parents[i]
        
    positive_data = Counts.iloc[:100]
    negative_data = Counts.iloc[100:]

    #Postive Probability
    Pos_proba = []
    for i, review in enumerate(df.text):
        prob = 1
        for j,ele in enumerate(Counts.iloc[i]):
            if ele==1:
                par = parents[j]
                if len(par)==0:
                    prob *= sum(positive_data[features[j]])/positive_data.shape[0]
                else:
                    temp = positive_data[positive_data[features[par[0]]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
        Pos_proba.append(prob)
                    
    #Negative Probability
    Neg_proba = []
    for i, review in enumerate(df.text):
        prob = 1
        for j,ele in enumerate(Counts.iloc[i]):
            if ele==1:
                par = parents[j]
                if len(par)==0:
                    prob *= sum(negative_data[features[j]])/negative_data.shape[0]
                else:
                    temp = negative_data[negative_data[features[par[0]]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
        Neg_proba.append(prob)
        
    Class = []
    for i, ele in enumerate(Pos_proba):
        if ele>Neg_proba[i]:
            Class.append(1)
        else:
            Class.append(0)
            
    print 'Accuracy of Naive Bayes with K2 (1 parent) =%f'% (sum(Class==df.stars)/len(df.stars))

##### Apply Naive Bayes Model with K2 (2 parents) #####
def Naive_Bayes_K2_2_parents(df,X,vectorizer):
    f = open('parents_22.txt')

    d = {}
    for line in f:
        node = line.strip().split(' ')[0]
        temp = line.strip().split(' ')[1:]
        parents = []
        if len(temp)>1:
            parents.append(temp[0].split('[')[1].split(',')[0])
            parents.append(temp[1].split(']')[0])
        elif len(temp[0])>2:
            parents.append(temp[0].split('[')[1].split(']')[0])       
                
        d[node] = parents
        
    features = vectorizer.get_feature_names()
    Counts = pd.DataFrame(X)
    Counts.columns = features
    positive_data = Counts.iloc[:100]
    negative_data = Counts.iloc[100:]
        
    #Positive Probability
    Pos_proba = []
    for i, review in enumerate(df.text):
        prob = 1
        for j,ele in enumerate(Counts.iloc[i]):
            if ele==1:
                par = d[str(j)]
                if len(par)==0:
                    prob *= sum(positive_data[features[j]])/positive_data.shape[0]
                elif len(par)==1:
                    temp = positive_data[positive_data[features[int(par[0])]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
                else:
                    temp = positive_data[positive_data[features[int(par[0])]]==1]
                    temp = temp[temp[features[int(par[1])]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
        Pos_proba.append(prob)
                    
    #Negative Probability
    Neg_proba = []
    for i, review in enumerate(df.text):
        prob = 1
        for j,ele in enumerate(Counts.iloc[i]):
            if ele==1:
                par = d[str(j)]
                if len(par)==0:
                    prob *= sum(negative_data[features[j]])/negative_data.shape[0]
                elif len(par)==1:
                    temp = negative_data[negative_data[features[int(par[0])]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
                else:
                    temp = negative_data[negative_data[features[int(par[0])]]==1]
                    temp = temp[temp[features[int(par[1])]]==1]
                    if temp.shape[0]>0:
                        prob *= sum(temp[features[j]])/temp.shape[0]
                
        Neg_proba.append(prob)
        
    Class = []
    for i, ele in enumerate(Pos_proba):
        if ele>Neg_proba[i]:
            Class.append(1)
        else:
            Class.append(0)
            
    print 'Accuracy of Naive Bayes with K2 (2 parents) =%f'% (sum(Class==df.stars)/len(df.stars))

if __name__ == '__main__':

    ##### Download data #####
    data = pd.read_csv('train.csv')

    ##### Clean and Tokenize the Data #####
    train = extract_positive_negative_reviews(data,100)
    train = clean(train)
    X, vectorizer = tokenize(train,200)
    #np.savetxt('array_yelp.txt', X)

    ##### Train, Predict and Measure Accuracy #####
    Naive_Bayes(train,X)
    Naive_Bayes_K2_1_parent(train,X,vectorizer)
    Naive_Bayes_K2_2_parents(train,X,vectorizer)

