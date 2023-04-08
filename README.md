# Tf-Idf_Search

My attempt at a TF-IDF search engine which makes use of the Pandas library to store the terms and numpy to do all the mathematical calculations

1. Uses SQLite for local vector space persistance
2. C++ Bindings used for document preprocessing to achieve significant speedup


tf-idf algorithm:
https://en.wikipedia.org/wiki/Tf%E2%80%93idf

future updates (in no particular order)
* port the Porter Stemmer algorithm to c++ and multithread it
* implement a frontend interface for the app


## How to run:
1. run `make` in the src/ path to create the shared library used by the python bindings
2. refer to `main.py` on how to use the `SearchEngine` class

note: if you want to change the max_docs parameter in the SearchEngine constructor and already have run the app on a source path, you will need to delete the searchUtils/ dir at that source path
