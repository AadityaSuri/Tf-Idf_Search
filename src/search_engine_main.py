
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
    stopwords_dict = Counter(stopwords.words('english'))

    file = open(filepath, 'r', errors='replace')
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
    # global tfcount
    # tfcount += 1
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
        

def tfidfMapBuilder(doclist):
    termset = set()

    for doc in doclist:
        termset.update(set(docPreProcessing(doc)))

    df_columns = doclist.copy()
    df_columns.append('df')
    print(df_columns)

    df = pd.DataFrame(0, index=list(termset), columns=df_columns)

    for doc in doclist:
        wordmap = tf(docPreProcessing(doc))
        for term in wordmap:
            df.at[term, doc] = wordmap[term]
            df.at[term, 'df'] += 1

    # print(df)

    N = len(doclist)
    # df['df'] = df['df'].apply(lambda x: math.log(N/x))

    return df

    # print(len(termset))

def mapToMatrix(df, N):
    df['df'] = df['df'].apply(lambda x: math.log(N/x))

    for index, row in df.iterrows():
        term_df = row['df']
        row = row.apply(lambda x: x * term_df)
    
    df = df.drop(columns=['df'], axis=1)

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

tfidfDataFrame = mapToMatrix(tfidfMapBuilder(doclist), len(doclist))


doc_tfidfMatrix = tfidfDataFrame.loc[:, tfidfDataFrame.columns != 'query.txt'].to_numpy().round(decimals=4)
query_vector = tfidfDataFrame.loc[:, "query.txt"].to_numpy().round(decimals=4)

scorelist = cosineSimilarityScore(query_vector, doc_tfidfMatrix)
scoremap = {}

for index, score in enumerate(scorelist):
    scoremap[tfidfDataFrame.columns[index + 1]] = score

scoremap = dict(sorted(scoremap.items(), key=lambda item:item[1]))

for doc in list(scoremap.items())[:5]:
    print(doc)



