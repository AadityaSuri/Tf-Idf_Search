from SearchEngine import SearchEngine
import pandas as pd
import sqlite3
import numpy as np

engine = SearchEngine("../testset/sci.space")

database = "../testset/sci.space/tfidfmap.db"
conn = sqlite3.connect(database)
tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
conn.close()

print(tfidfMap)