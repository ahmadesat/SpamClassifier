#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import io
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from pathlib import Path
path = os.path.join(Path().absolute(), "EmailSpamIdentifier/dataset")


# In[ ]:


def readFiles(path):
    for root,dirnames,filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root,filename)
            
            inBody = False
            lines= []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line== '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message
            
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename,message in readFiles(path):
        rows.append({"message" : message,"class" : classification})
        index.append(filename)
        
    return DataFrame(rows, index=index)

data = DataFrame({"message":[], "class":[]})

data = data.append(dataFrameFromDirectory(os.path.join(path, "spam"), 'spam'))
data = data.append(dataFrameFromDirectory(os.path.join(path, "ham"), 'not-spam'))


# In[ ]:


data.head()


# In[ ]:


vectorizer = CountVectorizer()
wordsCounter = vectorizer.fit_transform(data['message'].values) #splitting the message into words and counts how many times these words occur and then uses them in the predictor/classifier

classifier = MultinomialNB()
spamOrHam = data['class'].values
classifier.fit(wordsCounter, spamOrHam) #This will create a model which will predict whether a future email is spam or not spam.


# In[ ]:


examplesForTesting = ["Win your free Car now!!!", "Hello my friend, you ready for the game tomorrow?"]
                      
example_wordCounter = vectorizer.transform(examplesForTesting)

predictions = classifier.predict(example_wordCounter)

for n, v in zip(examplesForTesting, predictions):
    print("{} ========> {}".format(n, v))

