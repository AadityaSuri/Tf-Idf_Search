CC = g++
OBJ = tf.o
CFLAGS = -c -Wall -Iinclude

output: $(OBJ)
	$(CC) $(OBJ) -o $@

tf.o: tf.cpp stopwords.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o output


# c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) .cpp -o example$(python3-config --extension-suffix)