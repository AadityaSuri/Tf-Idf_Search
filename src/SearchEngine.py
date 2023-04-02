
import pandas as pd
import string
# import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import numpy as np
import os
import json
import copy
import sqlite3

# uncomment the following line if you are runnning this script for the first time or if you don't have the stopwords package

# import nltk
# nltk.download('stopwords')


class SearchEngine:
    def __init__(self, source_path) -> None:
        self.source_path = source_path
        self.database = source_path + "/searchUtils"


        if not os.path.exists(self.database):
            os.mkdir(self.database)
            self.__doclist = self.__fileCollector()
            self.__docNmap = {}
            self.__tfidfMapBuilder()
            conn = sqlite3.connect(self.database + '/tfidfmap.db')
            self.__tfidfMap.to_sql('tfidfmap', conn, if_exists='replace')
            conn.close()
            print("CHECKPOINT: tfidfmap.db created")
        else:
            conn = sqlite3.connect(self.database + '/tfidfmap.db')
            self.__tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn, index_col='index')
            conn.close()

            # load docNmap
            with open(self.database + '/docNmap.json', 'r') as f:
                self.__docNmap = json.load(f)
            
            self.__doclist = list(self.__docNmap.keys())




    def search(self, query):
        self.__addQuery(query)

        tfidfMatrix = self.__mapToMatrix()
        # tfidfMatrix.to_csv('tfidfMatrix.csv')
        # print(tfidfMatrix.index)

        doc_tfidfMatrix = tfidfMatrix.loc[:, tfidfMatrix.columns != 'query'].to_numpy().round(decimals=4)
        query_vector = tfidfMatrix.loc[:, "query"].to_numpy().round(decimals=4)

        scorelist = self.__cosineSimilarityScore(query_vector, doc_tfidfMatrix)

        scoremap = {}

        for index, score in enumerate(scorelist):
            scoremap[tfidfMatrix.columns[index]] = score

        scoremap = dict(sorted(scoremap.items(), key=lambda item:item[1]))



        for doc in list(scoremap.items())[:5]:
            print(doc)

        # print(scoremap)

    

    def __addQuery(self, query):
        queryfile = open('query.txt', 'w')
        queryfile.write(query)
        queryfile.close()

        tf_res = self.__tf(self.__docPreProcessing('query.txt'))
        query_tf = tf_res
        self.__docNmap['query'] = sum(tf_res.values())
        self.__doclist.append('query')

        self.__tfidfMap['query'] = 0

        for term in query_tf:
            if term in self.__tfidfMap.index:
                self.__tfidfMap.at[term, 'query'] = query_tf[term]
                self.__tfidfMap.at[term, 'df'] += 1
            else:
                new_row = pd.DataFrame({'df': [1], 'query': [query_tf[term]]}, index=[term])
                for column in self.__tfidfMap.columns:
                    if column != 'query':
                        new_row[column] = 0
                self.__tfidfMap = pd.concat([self.__tfidfMap, new_row])



    def __fileCollector(self):
        filelist = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".db"):
                    continue
                else:
                    filelist.append(os.path.join(root, file))

                if len(filelist) == 50:
                    return filelist

        return filelist
    
    
    def __docPreProcessing(self, filepath):
        stopwords_dict = Counter(stopwords.words('english'))
        ps = PorterStemmer()


        docTextList = []
        with open(filepath, 'r', errors='replace') as file:
            for line in file:
                for word in line.split():
                    word = word.translate(str.maketrans('', '', string.punctuation)).strip().lower()
                    if word not in stopwords_dict and word != '':
                        docTextList.append(ps.stem(word))

        return docTextList



    def __tf(self, wordlist):
        wordmap = {}
        
        for word in wordlist:
            if word in wordmap:
                wordmap[word] += 1
            else:
                wordmap[word] = 1
        
        return wordmap
    

    
    def __tfidfMapBuilder(self):
        termset = set()

        for doc in self.__doclist:
            tf_res = self.__tf(self.__docPreProcessing(doc))
            self.__docNmap[doc] = sum(tf_res.values())
            termset.update(set(tf_res.keys()))

        with open(self.database + '/docNmap.json', 'w') as f:
            json.dump(self.__docNmap, f, indent=4)

        df_columns = ['df']
        df_columns.extend(self.__doclist)
       
        self.__tfidfMap = pd.DataFrame(0, index=list(termset), columns=df_columns)

        for doc in self.__doclist:
            wordmap = self.__tf(self.__docPreProcessing(doc))
            for term in wordmap:
                self.__tfidfMap.at[term, doc] = wordmap[term]
                self.__tfidfMap.at[term, 'df'] += 1




    def __mapToMatrix(self):
        matrix = copy.copy(self.__tfidfMap)

        N = len(self.__doclist)
        matrix['df'] = matrix['df'].apply(lambda x: math.log((N + 1)/(x + 1)) + 1)

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
    



