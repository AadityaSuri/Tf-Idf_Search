from SearchEngine import SearchEngine
import time
# import cppbindings

constructorStartTime = time.process_time()
engine = SearchEngine(r"../testset/sci.space", max_docs=200)
print("Time taken to construct: ", time.process_time() - constructorStartTime)

# database = r'../testset/sci.space/tfidfmap.db'
# conn = sqlite3.connect(database)
# tfidfMap = pd.read_sql_query("SELECT * FROM tfidfmap", conn)
# conn.close()

# engine.printmap()

seachStartTime = time.process_time()
engine.search("EM radiation in the solar system")
print("Time taken to search: ", time.process_time() - seachStartTime)

# print(cppbindings.docPreProcessing(r'../testset/sci.space/60151'))
