CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC
LDLIBS = $(shell python3-config --ldflags)
LDFLAGS = -shared
INCLUDES = -Iinclude $(shell python3 -m pybind11 --includes)

BINDING_SRC = bindings/binding.cpp
BINDING_OBJ = $(patsubst %.cpp,%.o,$(BINDING_SRC))

DOC_PREPROCESSING_SRC = docPreProcessing.cpp 
TF_SRC = tf.cpp
DOC_PREPROCESSING_OBJ = $(patsubst %.cpp,%.o,$(DOC_PREPROCESSING_SRC))
TF_OBJ = $(patsubst %.cpp,%.o,$(TF_SRC))

cppbindings = cppbindings$(shell python3-config --extension-suffix)

all: $(cppbindings)

$(cppbindings): $(BINDING_OBJ) $(DOC_PREPROCESSING_OBJ) $(TF_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BINDING_OBJ): $(BINDING_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(DOC_PREPROCESSING_OBJ): $(DOC_PREPROCESSING_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TF_OBJ): $(TF_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean-all:
	rm -f $(cppbindings) $(BINDING_OBJ) $(DOC_PREPROCESSING_OBJ) $(TF_OBJ)

clean:
	rm -f $(BINDING_OBJ) $(DOC_PREPROCESSING_OBJ) $(TF_OBJ)
