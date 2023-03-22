# %%
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
import math
import numpy as np
import os



def docPreProcessing(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    
    print(filepath)

    stopwords_dict = Counter(stopwords.words('english'))
    
    
    doctext = ""
    for line in lines:
        line = line.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        line = ' '.join([word for word in line.split() if word not in stopwords_dict])
        doctext += line
        
    doctextList = doctext.split()
    return doctextList
        

def tf(wordlist):
    wordmap = {}
    
    for word in wordlist:
        if word in wordmap:
            wordmap[word] += 1
        else:
            wordmap[word] = 1
            
    
    numWords = len(wordlist)
    
    for word in wordmap:
        wordmap[word] = wordmap[word] / numWords
    
    return wordmap
        


#make this faster
def idf(doclist):
    idfmap = {}
    
    for doc in doclist:
        wordmap = tf(docPreProcessing(doc))
        
        for term in wordmap:
            if term in idfmap:
                idfmap[term] += 1
            else:
                idfmap[term] = 1
                
    N = len(doclist)
                
            
    for term in idfmap:
        idfmap[term] = math.log(N/idfmap[term], 2)
        
    return idfmap
        

def docMatrixBuilder(doclist, tf, idf):
    idfmap = idf(doclist)
    terms = list(idfmap)
    df = pd.DataFrame(0, index = terms, columns = doclist)
    
    for doc in doclist:
        wordmap = tf(docPreProcessing(doc))
        
        for word in wordmap:
            df.loc[word, doc] = wordmap[word]
        
        for term in idfmap:
            df.loc[term, doc] *= idfmap[term]            
    
    
    return df

def cosineSimilarityScore(q, D):
    q_mag = np.sqrt(q.dot(q))
    
    qT = np.reshape(q, (1, q.shape[0]))
    qTD = np.matmul(qT, D).reshape((D.shape[1], ))
    
    D_mags = np.sqrt(np.sum(D*D, axis=0))
    divisors = q_mag * D_mags
    cos_thetas = np.divide(qTD, divisors)
    
    cos_thetas = np.clip(cos_thetas, -1, 1)
    
    scores = np.arccos(cos_thetas)
    
    return scores
    
        

def fileCollector(path):
    filelist = []
    
    for (root, dirs, files) in os.walk(path, topdown=True):
        for file in files:
            filelist.append(os.path.join(root, file))
            
            if len(filelist) == 50:
                return filelist

    return filelist

doclist = []

doclist.append('query.txt')

doclist.extend(fileCollector(r'../testset/sci.space'))


# i = 1
# for i in range(1, 7):
#      doclist.append('sample' + str(i) + '.txt')

tfidfDataframe = docMatrixBuilder(doclist, tf, idf)

doc_tfidfMatrix = tfidfDataframe.loc[:, tfidfDataframe.columns != 'query.txt'].to_numpy().round(decimals=4)
query_vector = tfidfDataframe.loc[:, "query.txt"].to_numpy().round(decimals=4)

scorelist = cosineSimilarityScore(query_vector, doc_tfidfMatrix).tolist()

scoremap = {}


for index, score in enumerate(scorelist):
    scoremap[tfidfDataframe.columns[index + 1]] = score

scoremap = dict(sorted(scoremap.items(), key=lambda item:item[1]))

for doc in list(scoremap.items())[:5]:
    print(doc)


        


