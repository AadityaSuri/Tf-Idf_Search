
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
import math
import numpy as np
import os
import json
import copy
import sqlite3

class SearchEngine:
    def __init__(self, source_path) -> None:
        self.source_path = source_path
        database = source_path + "/tfidfmap.db"


        if not os.path.exists(database):
            self.__doclist = self.__fileCollector()
            self.__docNmap = {}
            self.__tfidfMapBuilder()
            # database = source_path + "/tfidfmap.db"
            conn = sqlite3.connect(database)
            self.__tfidfMap.to_sql('tfidfmap', conn, if_exists='replace')
            conn.close()
            print("CHECKPOINT: tfidfmap.db created")
        else:
            # database = source_path + "/tfidfmap.db"
            conn = sqlite3.connect(database)
            self.__tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn, index_col='index')
            conn.close()
            # load docNmap
            with open('docNmap.json', 'r') as f:
                self.__docNmap = json.load(f)
            
            self.__doclist = list(self.__docNmap.keys())
            print("CHECKPOINT: tfidfmap.db loaded")

        # read tfidfmap.db and calculate __docllist and __docNmap



    def search(self, query):
        self.__addQuery(query)
        # tfidfMap = addQuery_res
        # docNmap = addQuery_res[1]

        tfidfMatrix = self.__mapToMatrix()
        tfidfMatrix.to_csv('tfidfMatrix.csv')

        # doc_tfidfMatrix = tfidfMatrix.loc[:, tfidfMap.columns != 'query'].to_numpy().round(decimals=4)
        # query_vector = tfidfMatrix.loc[:, "query"].to_numpy().round(decimals=4)

        # scorelist = self.__cosineSimilarityScore(query_vector, doc_tfidfMatrix)

        # return scorelist
    


    def __addQuery(self, query):


        queryfile = open('query.txt', 'w')
        queryfile.write(query)
        queryfile.close()

        tf_res = self.__tf(self.__docPreProcessing('query.txt'))
        query_tf = tf_res
        self.__docNmap['query'] = sum(tf_res.values())
        self.__doclist.append('query')
        # docNmap = tf_res[1]

        self.__tfidfMap['query'] = 0

        for term in query_tf:
            if term in self.__tfidfMap.index:
                self.__tfidfMap.at[term, 'query'] = query_tf[term]
                self.__tfidfMap.at[term, 'df'] += 1
            else:
                new_row = pd.DataFrame(data={'df': 1, 'query': query_tf[term]}, index=[term])
                self.__tfidfMap = pd.concat([self.__tfidfMap, new_row])

        # return tfidfMap
    

    
    def printmap(self):
        database = self.source_path + "/tfidfmap.db"
        conn = sqlite3.connect(database)
        df = pd.read_sql_query("SELECT * FROM tfidfmap", conn, index_col='index')
        conn.close()
        self.__addQuery("EM Radiation in the orion belt")
        df.to_csv('tfidfmap.csv')
        print(df)
        # print(df.columns)



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
                
        
        # numWords = len(wordlist)
        
        # for word in wordmap:
        #     wordmap[word] = wordmap[word] / numWords
        
        return wordmap
    

    
    def __tfidfMapBuilder(self):
        termset = set()
        # docNmap = {}

        for doc in self.__doclist:
            tf_res = self.__tf(self.__docPreProcessing(doc))
            self.__docNmap[doc] = sum(tf_res.values())
            termset.update(set(tf_res.keys()))

        with open('docNmap.json', 'w') as f:
            json.dump(self.__docNmap, f, indent=4)

        df_columns = ['df']
        df_columns.extend(self.__doclist)
       
        self.__tfidfMap = pd.DataFrame(0, index=list(termset), columns=df_columns)

        for doc in self.__doclist:
            wordmap = self.__tf(self.__docPreProcessing(doc))
            for term in wordmap:
                self.__tfidfMap.at[term, doc] = wordmap[term]
                self.__tfidfMap.at[term, 'df'] += 1

        # return df




    def __mapToMatrix(self):
        matrix = copy.copy(self.__tfidfMap)

        N = len(self.__doclist)
        matrix['df'] = matrix['df'].apply(lambda x: math.log(N/x))

        for col in matrix.columns:
            if col != 'df':
                matrix[col] = matrix[col].apply(lambda x: x/self.__docNmap[col])

        for index, row in matrix.iterrows():
            term_df = row['df']
            row = row.apply(lambda x: x * term_df)
        
        matrix = matrix.drop(columns=['df'], axis=1)

        return matrix
    
    

    
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



