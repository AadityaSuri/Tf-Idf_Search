from SearchEngine import SearchEngine

engine = SearchEngine(r"../testset/sci.space")

# database = r'../testset/sci.space/tfidfmap.db'
# conn = sqlite3.connect(database)
# tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
# conn.close()

# engine.printmap()
engine.search("Moon Landing is a hoax")
