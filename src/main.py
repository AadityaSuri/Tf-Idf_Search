from SearchEngine import SearchEngine
import time

engine = SearchEngine(r"../testset/sci.space", max_docs=200)

# database = r'../testset/sci.space/tfidfmap.db'
# conn = sqlite3.connect(database)
# tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
# conn.close()

# engine.printmap()

seachStartTime = time.process_time()
engine.search("EM radiation in the solar system")
print("Time taken to search: ", time.process_time() - seachStartTime)
