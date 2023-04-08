# CC = g++
# OBJ = tf.o
# CFLAGS = -c -Wall -Iinclude

# output: $(OBJ)
# 	$(CC) $(OBJ) -o $@

# tf.o: tf.cpp stopwords.h
# 	$(CC) $(CFLAGS) $< -o $@

# clean:
# 	rm -f *.o output


# # c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) .cpp -o example$(python3-config --extension-suffix)

CXX = c++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC
LDLIBS = $(shell python3-config --ldflags)
LDFLAGS = -shared
INCLUDES = -Iinclude $(shell python3 -m pybind11 --includes)

BINDING_SRC = bindings/binding.cpp
BINDING_OBJ = $(patsubst %.cpp,%.o,$(BINDING_SRC))

DOC_PREPROCESSING_SRC = docPreProcessing.cpp
DOC_PREPROCESSING_OBJ = $(patsubst %.cpp,%.o,$(DOC_PREPROCESSING_SRC))

cppbindings = cppbindings$(shell python3-config --extension-suffix)

all: $(cppbindings)

$(cppbindings): $(BINDING_OBJ) $(DOC_PREPROCESSING_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BINDING_OBJ): $(BINDING_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(DOC_PREPROCESSING_OBJ): $(DOC_PREPROCESSING_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(cppbindings) $(BINDING_OBJ) $(DOC_PREPROCESSING_OBJ)
