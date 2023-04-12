#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "stopwords.h"
#include "cppfuncs.h"


std::vector<std::string> docPreProcessing(std::string filepath) {

    // Create a map of stop words for o(1) lookup
    std::unordered_map<std::string, int> stop_word_counter;
    for (std::string word : stop_words) {
        stop_word_counter[word] = 0;
    }

    std::ifstream file(filepath);
    std::string word;

    std::vector<std::string> docTextList;
    while (file >> word) {
        // Convert the word to lowercase
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        // Replace non-ASCII characters with a question mark '?'
        std::string sanitized;
        for (char ch : word) {
            if (static_cast<unsigned char>(ch) > 127) {
                sanitized += '?';
            } else {
                sanitized += ch;
            }
        }

        // Remove punctuation and check if the sanitized word is a stop word
        sanitized.erase(std::remove_if(sanitized.begin(), sanitized.end(),
                                       [](char c) { return std::ispunct(c); }), sanitized.end());

        // Add the word to the list if it is not a stop word
        if (stop_word_counter.find(sanitized) == stop_word_counter.end() && sanitized.length() > 0) {
            docTextList.push_back(sanitized);
        }
    }
    
    return docTextList;
}



