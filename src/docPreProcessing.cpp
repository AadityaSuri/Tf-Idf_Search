#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "include/stopwords.h"
#include "include/cppfuncs.h"

std::vector<std::string> docPreProcessing(std::string filepath) {

    std::fstream file;
    file.open(filepath, std::ios::in);
    std::string word;

    std::vector<std::string> words;
    while (file >> word) {
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        std::string sanitized;
        std::remove_copy_if(word.begin(), word.end(), std::back_inserter(sanitized), 
                            [](char c) { return std::ispunct(c); });

        if (sanitized.length() > 0) {
            words.push_back(sanitized);
        }
    }

    return words;
}