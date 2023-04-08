#include <string>
#include <unordered_map>
#include "include/cppfuncs.h"

std::unordered_map<std::string, int> tf(std::vector<std::string> docTextList) {
    std::unordered_map<std::string, int> term_frequency;
    for (std::string word : docTextList) {
        term_frequency[word]++;
    }
    return term_frequency;
}