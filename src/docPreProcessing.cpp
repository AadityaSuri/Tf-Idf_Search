#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "include/stopwords.h"
#include "include/cppfuncs.h"


std::vector<std::string> docPreProcessing(std::string filepath) {

        std::unordered_map<std::string, int> stop_word_counter;
        for (std::string word : stop_words) {
            stop_word_counter[word] = 0;
        }

        std::fstream file;
        file.open(filepath, std::ios::in);
        std::string word;

        std::vector<std::string> docTextList;
        while (file >> word) {
            std::transform(word.begin(), word.end(), word.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            std::string sanitized;
            std::remove_copy_if(word.begin(), word.end(), std::back_inserter(sanitized), 
                                [](char c) { return std::ispunct(c); });

            if (stop_word_counter.find(sanitized) == stop_word_counter.end() && sanitized.length() > 0) {
                docTextList.push_back(sanitized);
            }
        }
        
        return docTextList;
}


// #include <fstream>
// #include <string>
// #include <unordered_map>
// #include <vector>
// #include <algorithm>
// #include <cctype>

// std::vector<std::string> docPreProcessing(std::string filepath) {
//     // Define stop_words and stop_word_counter as before...

//     std::unordered_map<std::string, int> stop_word_counter;
//     for (std::string word : stop_words) {
//         stop_word_counter[word] = 0;
//     }

//     std::ifstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         return std::vector<std::string>();
//     }

//     std::string word;

//     std::vector<std::string> docTextList;
//     while (file >> word) {
//         try {
//             std::transform(word.begin(), word.end(), word.begin(),
//                            [](unsigned char c) { return std::tolower(c); });

//             std::string sanitized;
//             std::remove_copy_if(word.begin(), word.end(), std::back_inserter(sanitized),
//                                 [](char c) { return std::ispunct(c); });

//             if (stop_word_counter.find(sanitized) == stop_word_counter.end() && sanitized.length() > 0) {
//                 docTextList.push_back(sanitized);
//             }
//         } catch (std::exception& e) {
//             // Replace invalid byte sequence with placeholder character
//             std::replace_if(word.begin(), word.end(),
//                             [](unsigned char c) { return c < 32 || c == 127; }, '?');

//             std::transform(word.begin(), word.end(), word.begin(),
//                            [](unsigned char c) { return std::tolower(c); });

//             std::string sanitized;
//             std::remove_copy_if(word.begin(), word.end(), std::back_inserter(sanitized),
//                                 [](char c) { return std::ispunct(c); });

//             if (stop_word_counter.find(sanitized) == stop_word_counter.end() && sanitized.length() > 0) {
//                 docTextList.push_back(sanitized);
//             }
//         }
//     }

//     return docTextList;
// }
