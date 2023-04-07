CC = g++
OBJ = tf.o
CFLAGS = -c -Wall -Iinclude

output: $(OBJ)
	$(CC) $(OBJ) -o $@

tf.o: tf.cpp stopwords.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o output
