from SearchEngine import SearchEngine
import pandas as pd
import sqlite3
import numpy as np
import os

engine = SearchEngine(r'../testset/sci.space')

# database = r'../testset/sci.space/tfidfmap.db'
# conn = sqlite3.connect(database)
# tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
# conn.close()

engine.printmap()

