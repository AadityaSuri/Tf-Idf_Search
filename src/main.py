from SearchEngine import SearchEngine
import time

engine = SearchEngine(r"../testset/sci.space", max_docs=200)

top_n = engine.search("EM radiation in the solar system")

print(top_n)

