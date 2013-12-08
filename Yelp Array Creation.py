# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647,0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (15, 11)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

# <codecell>

train = pd.read_csv('D:/Kaggle/Yelp Recruiting Competition/train.csv')

# <codecell>

positive = train[train.stars==5]
negative = train[train.stars==1]

# <codecell>

positive = positive.iloc[:100]
negative = negative.iloc[:100]
positive = positive[['text','stars']]
negative = negative[['text','stars']]
pos_reviews = list(positive.text)
nega_reviews = list(negative.text)

# <codecell>

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

# <codecell>

pos_df = clean(positive)
neg_df = clean(negative)

# <codecell>

pos_df.stars = pos_df.stars.replace(5,1)
neg_df.stars = neg_df.stars.replace(1,0)

# <codecell>

train_df = pd.concat([pos_df,neg_df])

# <codecell>

#TOKENIZE
print 'Start Tokenizing Using Tf-Idf'
vectorizer = CountVectorizer(min_df=10,stop_words='english',binary = True, max_features = 200)
X = vectorizer.fit_transform(train_df.text)
X = X.toarray()

# <codecell>

#np.savetxt('array_yelp.txt', X)

# <codecell>

#Naive Bayes
clf = MultinomialNB(fit_prior=False)
clf.fit(X, list(train_df.stars))
#print len(train_df.stars)
print 'Accuracy to beat =%f'% (sum(clf.predict(X)==train_df.stars)/len(train_df.stars))

# <codecell>

#Augmented Naive Bayes with K2-Graphical Structure
parents = pickle.load(open('parents.txt'))
features = vectorizer.get_feature_names()
Counts = pd.DataFrame(X)
Counts.columns = features

#Dictionnary of (Key, Values) = (Word, Parents)
d = {}
for i,ele in enumerate(features):
    d[ele] = parents[i]

# <codecell>

positive_data = Counts.iloc[:100]
negative_data = Counts.iloc[100:]

# <codecell>

#Postive Probability
Pos_proba = []
for i, review in enumerate(train_df.text):
    prob = 1
    for j,ele in enumerate(Counts.iloc[i]):
        if ele==1:
            par = parents[j]
            if len(par)==0:
                prob *= sum(positive_data[features[j]])/len(positive_data.stars)
            else:
                temp = positive_data[positive_data[features[par[0]]]==1]
                if len(temp.stars)>0:
                    prob *= sum(temp[features[j]])/len(temp.stars)
    Pos_proba.append(prob)
                

# <codecell>

#Negative Probability
Neg_proba = []
for i, review in enumerate(train_df.text):
    prob = 1
    for j,ele in enumerate(Counts.iloc[i]):
        if ele==1:
            par = parents[j]
            if len(par)==0:
                prob *= sum(negative_data[features[j]])/len(negative_data.stars)
            else:
                temp = negative_data[negative_data[features[par[0]]]==1]
                if len(temp.stars)>0:
                    prob *= sum(temp[features[j]])/len(temp.stars)
    Neg_proba.append(prob)

# <codecell>

Class = []
for i, ele in enumerate(Pos_proba):
    if ele>Neg_proba[i]:
        Class.append(1)
    else:
        Class.append(0)

# <codecell>

sum(Class==train_df.stars)/len(train_df.stars)

# <codecell>


