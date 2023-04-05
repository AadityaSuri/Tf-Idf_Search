
import pandas as pd
import string
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



# SearchEngine main class
class SearchEngine:

    # constructor
    def __init__(self, source_path: 'str') -> None:
        self.source_path = source_path  # path to the source directory at which all documents are to be searched
        self.database = source_path + "/searchUtils"


        # create tfidfMatrix for all terms present in the source directory if it doesn't exist
        if not os.path.exists(self.database):

            # create database directory
            os.mkdir(self.database)

            # create tfidfmap.db
            self.__doclist = self.__fileCollector()  # recursively list all documents in the source directory
            self.__docNmap = {}                      # map of document name to number of terms in the document
            self.__tfidfMapBuilder()

            # save tfidfmap for searching queries
            conn = sqlite3.connect(self.database + '/tfidfmap.db')
            self.__tfidfMap.to_sql('tfidfmap', conn, if_exists='replace')
            conn.close()
            print("CHECKPOINT: tfidfmap.db created")

        # load tfidfmap for searching queries from existing database
        else:
            # load tfidfmap
            conn = sqlite3.connect(self.database + '/tfidfmap.db')
            self.__tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn, index_col='index')
            conn.close()

            # load docNmap
            with open(self.database + '/docNmap.json', 'r') as f:
                self.__docNmap = json.load(f)
            
            self.__doclist = list(self.__docNmap.keys())



    # print top 5 most relevant documents for the given query
    def search(self, query: 'str') -> None:
        self.__addQuery(query)  # add query to the tfidfMap

        tfidfMatrix = self.__mapToMatrix()  # convert tfidfMap to tfidfMatrix with the query added
        # tfidfMatrix.to_csv('tfidfMatrix.csv')
        # print(tfidfMatrix.index)


        # extract the document and query vectors from the matrix
        doc_tfidfMatrix = tfidfMatrix.loc[:, tfidfMatrix.columns != 'query'].to_numpy().round(decimals=4)
        query_vector = tfidfMatrix.loc[:, "query"].to_numpy().round(decimals=4)

        scorelist = self.__cosineSimilarityScore(query_vector, doc_tfidfMatrix)

        scoremap = {}

        for index, score in enumerate(scorelist):
            scoremap[tfidfMatrix.columns[index]] = score

        scoremap = dict(sorted(scoremap.items(), key=lambda item:item[1]))

        for doc in list(scoremap.items())[:5]:
            print(doc)

    

    # add query to the tfidfMap
    def __addQuery(self, query : 'str') -> None:

        # write query to a file (inefficient, but works. need to change this)
        queryfile = open('query.txt', 'w')
        queryfile.write(query)
        queryfile.close()

        # perform document preprocessing on the query
        tf_res = self.__tf(self.__docPreProcessing('query.txt'))
        query_tf = tf_res

        # add query to the tfidfMap and update the df of each term
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



    # recusively collect all files in the source directory
    def __fileCollector(self) -> 'list':
        filelist = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(".json") or file.endswith(".db"):  # ignore json and db files since they are for internal use, need to change this
                    continue
                else:
                    filelist.append(os.path.join(root, file))

                if len(filelist) == 50:  # cap the number of documents to 50 for now, need to change this
                    return filelist

        return filelist
    
    
    # perform some preprocessing on the document
    # 1. remove any stopwords from the document (using nltk stopwords package) 
    # 2. remove any punctuation from the document
    # 3. convert all words to lowercase
    # 4. perform stemming on the document (using nltk PorterStemmer)
    def __docPreProcessing(self, filepath: 'str') -> 'list':
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



    # calculate the tf of each term in the document
    # tf = (number of times term t appears in a document) / (total number of (non unique) terms in the document)

    # kinda slow, maybe optimize this using c++ bindings
    def __tf(self, wordlist : 'list') -> 'dict':
        wordmap = {}
        
        for word in wordlist:
            if word in wordmap:
                wordmap[word] += 1
            else:
                wordmap[word] = 1
        
        return wordmap
    



    # Build the tfidfMap from the doclist 
    def __tfidfMapBuilder(self) -> None:
        termset = set()

        for doc in self.__doclist:
            tf_res = self.__tf(self.__docPreProcessing(doc))
            self.__docNmap[doc] = sum(tf_res.values())
            termset.update(set(tf_res.keys()))

        # save docNmap to json file
        with open(self.database + '/docNmap.json', 'w') as f:
            json.dump(self.__docNmap, f, indent=4)

        df_columns = ['df']
        df_columns.extend(self.__doclist)
       
        self.__tfidfMap = pd.DataFrame(0, index=list(termset), columns=df_columns)


        # can parallelize this for loop. at tf for all parallely but block at df
        for doc in self.__doclist:
            wordmap = self.__tf(self.__docPreProcessing(doc))
            for term in wordmap:
                self.__tfidfMap.at[term, doc] = wordmap[term]
                self.__tfidfMap.at[term, 'df'] += 1




    # map the tfidfMap to a matrix, remove the df column and normalize the matrix
    def __mapToMatrix(self) -> 'pd.DataFrame':
        matrix = copy.copy(self.__tfidfMap)

        N = len(self.__doclist)
        # df = number of documents that contain the term t (document frequency) 
        matrix['df'] = matrix['df'].apply(lambda x: math.log((N + 1)/(x + 1)) + 1)  # parallelize the apply function

        for col in matrix.columns:
            if col != 'df':
                matrix[col] = matrix[col].apply(lambda x: x/self.__docNmap[col])  # parallelize the apply function

        # parallelize this for loop
        for index, row in matrix.iterrows():
            term_df = row['df']
            row = row.apply(lambda x: x * term_df) # parallelize the apply function
        
        matrix = matrix.drop(columns=['df'], axis=1) # remove the df column

        return matrix
    
    

    # calculate the cosine similarity score between the query and each document
    # cos(theta) = (qT * D) / (||q|| * ||D||)
    # theta = arccos(cos(theta))
    # minimize theta to find the most similar document
    def __cosineSimilarityScore(self, q, D) -> 'np.array':
        q_mag = np.sqrt(q.dot(q))
        
        qT = np.reshape(q, (1, q.shape[0]))
        qTD = np.matmul(qT, D).reshape((D.shape[1], ))
        
        D_mags = np.sqrt(np.sum(D*D, axis=0))
        divisors = q_mag * D_mags
        cos_thetas = np.divide(qTD, divisors)
        
        cos_thetas = np.clip(cos_thetas, -1, 1)  # clip values to [-1, 1] to avoid nan values
        
        scores = np.arccos(cos_thetas)
        
        return scores
    



