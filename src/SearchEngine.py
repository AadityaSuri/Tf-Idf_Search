
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

        tfidfMap = self.__tfidfMapBuilder()[0]
        database = source_path + "/tfidfmap.db"
        conn = sqlite3.connect(database)
        tfidfMap.to_sql('tfidfmap', conn, if_exists='replace')
        conn.close()


    def search(self, query):
        addQuery_res = self.__addQuery(query)
        tfidfMap = addQuery_res[0]
        docNmap = addQuery_res[1]

        tfidfMatrix = self.__mapToMatrix(tfidfMap, len(self.doclist) + 1, docNmap).to_numpy().round(decimals=4)
        print(tfidfMatrix)

        # doc_tfidfMatrix = tfidfMatrix.loc[:, tfidfMap.columns != 'query'].to_numpy().round(decimals=4)
        # query_vector = tfidfMatrix.loc[:, "query"].to_numpy().round(decimals=4)

        # scorelist = self.__cosineSimilarityScore(query_vector, doc_tfidfMatrix)

        # return scorelist
    


    def __addQuery(self, query):
        database = self.source_path + "/tfidfmap.db"
        conn = sqlite3.connect(database)
        tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn, index_col='index')
        conn.close()

        queryfile = open('query.txt', 'w')
        queryfile.write(query)
        queryfile.close()

        tf_res = self.__tf(self.__docPreProcessing('query.txt'))
        query_tf = tf_res[0]
        docNmap = tf_res[1]

        tfidfMap['query'] = 0

        for term in query_tf:
            if term in tfidfMap.index:
                tfidfMap.at[term, 'query'] = query_tf[term]
                tfidfMap.at[term, 'df'] += 1
            else:
                new_row = pd.DataFrame(data={'df': 1, 'query': query_tf[term]}, index=[term])
                tfidfMap = pd.concat([tfidfMap, new_row])

        return (tfidfMap, docNmap)
    
    def printmap(self):
        df = self.__addQuery("EM Radiation in the orion belt")
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
                
        
        numWords = len(wordlist)
        
        # for word in wordmap:
        #     wordmap[word] = wordmap[word] / numWords
        
        return (wordmap, len(wordlist))
    

    
    def __tfidfMapBuilder(self):
        termset = set()
        docNmap = {}

        for doc in self.doclist:
            tf_res = self.__tf(self.__docPreProcessing(doc))
            docNmap[doc] = tf_res[1]
            termset.update(set(tf_res[0].keys()))

        df_columns = ['df']
        df_columns.extend(self.doclist)
       
        df = pd.DataFrame(0, index=list(termset), columns=df_columns)

        for doc in self.doclist:
            wordmap = self.__tf(self.__docPreProcessing(doc))[0]
            for term in wordmap:
                df.at[term, doc] = wordmap[term]
                df.at[term, 'df'] += 1

        return (df, docNmap)




    def __mapToMatrix(self, df, N, docNmap):
        df['df'] = df['df'].apply(lambda x: math.log(N/x))

        for col in df.columns:
            if col != 'df':
                df[col] = df[col].apply(lambda x: x/docNmap[col])

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



