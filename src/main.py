from SearchEngine import SearchEngine
import time

start = time.time()
engine = SearchEngine(r"../testset/sci.space")
print("Time taken to initialize SearchEngine: ", time.time() - start)

start = time.time()
top_n = engine.search("EM radiation in the solar system")
print("Time taken to search: ", time.time() - start)

print(top_n)
