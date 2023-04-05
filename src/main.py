from SearchEngine import SearchEngine

engine = SearchEngine(r"../testset/sci.space", max_docs=50)

# database = r'../testset/sci.space/tfidfmap.db'
# conn = sqlite3.connect(database)
# tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
# conn.close()

# engine.printmap()
engine.search("EM radiation in the solar system")
