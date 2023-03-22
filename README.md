# Tf-Idf_Search

My attempt at a TF-IDF search engine which makes use of the Pandas library to store the terms and numpy to do all the mathematical calculations

future changes:

1. Make the idf function faster by changing df values "on the fly"
2. Add some persistence using a database instead of pulling the entire vector space in memory
3. Port the codebase to C++ and introduce multithreading using xtensor to replace numpy
