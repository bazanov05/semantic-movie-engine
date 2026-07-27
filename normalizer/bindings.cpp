#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector and std::string to Python lists and strings
#include "normalizer.hpp"

namespace py = pybind11;

// (compiled file, C++ obj)
PYBIND11_MODULE(normalizer_python, m) {
    py::class_<TextNormalizer>(m, "TextNormalizer")
        // provide types for constructor since there is not pointer to it 
        .def(py::init<const std::unordered_set<std::string>>(), py::arg("stop_words"))
        // no types are needed since there is a pointer to method 
        .def("clean", &TextNormalizer::clean, py::arg("dirty_text"));
}