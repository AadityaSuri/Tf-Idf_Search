#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cppfuncs.h"

namespace py = pybind11;

PYBIND11_MODULE(cppbindings, m) {
    m.doc() = "C++ bindings for docPreProcessing() using PyBind11"; // optional module docstring

    m.def("docPreProcessing", &docPreProcessing, 
        "A function that preprocesses a documment, sanitizes it and retursn a vector of words");
}


