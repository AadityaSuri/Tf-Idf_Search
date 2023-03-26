
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
import math
import numpy as np
import os
import sqlite3

class SearchEngine:
    def __init__(self, source_path) -> None:
        self.source_path = source_path
        self.doclist = self.__fileCollector()

        tfidfMap = self.__tfidfMapBuilder()
        database = source_path + "/tfidfmap.db"
        conn = sqlite3.connect(database)
        tfidfMap.to_sql('tfidfmap', conn, if_exists='replace')
        conn.close()


    def __fileCollector(self):
        filelist = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                filelist.append(os.path.join(root, file))

                if len(filelist) == 50:
                    return filelist

        return filelist
    


    def __docPreProcessing(self, filepath):
        stopwords_dict = Counter(stopwords.words('english'))

        file = open(filepath, 'r', errors='replace')
        lines = file.readlines()
        
        stopwords_dict = Counter(stopwords.words('english'))
        
        
        doctext = ""
        for line in lines:
            line = line.translate(str.maketrans('', '', string.punctuation)).strip().lower()
            line = ' '.join([word for word in line.split() if word not in stopwords_dict])
            doctext += line

        file.close()
            
        doctextList = doctext.split()
        return doctextList
    


    def __tf(self, wordlist):
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
    

    
    def __tfidfMapBuilder(self):
        termset = set()

        for doc in self.doclist:
            termset.update(set(self.__docPreProcessing(doc)))

        df_columns = self.doclist.copy()
        df_columns.append('df')
        # print(df_columns)

        df = pd.DataFrame(0, index=list(termset), columns=df_columns)

        for doc in self.doclist:
            wordmap = self.__tf(self.__docPreProcessing(doc))
            for term in wordmap:
                df.at[term, doc] = wordmap[term]
                df.at[term, 'df'] += 1

        return df




    def __mapToMatrix(self, df, N):
        df['df'] = df['df'].apply(lambda x: math.log(N/x))

        for index, row in df.iterrows():
            term_df = row['df']
            row = row.apply(lambda x: x * term_df)
        
        df = df.drop(columns=['df'], axis=1)

        return df
    

    
    def __cosineSimilarityScore(self, q, D):
        q_mag = np.sqrt(q.dot(q))
        
        qT = np.reshape(q, (1, q.shape[0]))
        qTD = np.matmul(qT, D).reshape((D.shape[1], ))
        
        D_mags = np.sqrt(np.sum(D*D, axis=0))
        divisors = q_mag * D_mags
        cos_thetas = np.divide(qTD, divisors)
        
        cos_thetas = np.clip(cos_thetas, -1, 1)
        
        scores = np.arccos(cos_thetas)
        
        return scores
    

    
# doclist = []

# doclist.append('query.txt')

# doclist.extend(fileCollector(r'../testset/sci.space'))

# tfidfDataFrame = mapToMatrix(tfidfMapBuilder(doclist), len(doclist))


# doc_tfidfMatrix = tfidfDataFrame.loc[:, tfidfDataFrame.columns != 'query.txt'].to_numpy().round(decimals=4)
# query_vector = tfidfDataFrame.loc[:, "query.txt"].to_numpy().round(decimals=4)

# scorelist = cosineSimilarityScore(query_vector, doc_tfidfMatrix)
# scoremap = {}

# for index, score in enumerate(scorelist):
#     scoremap[tfidfDataFrame.columns[index + 1]] = score

# scoremap = dict(sorted(scoremap.items(), key=lambda item:item[1]))

# for doc in list(scoremap.items())[:5]:
#     print(doc)



