#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "stopwords.h"

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


std::unordered_map<std::string, double> tf(std::string filepath) {

        std::unordered_map<std::string, int> stop_word_counter;
        for (std::string word : stop_words) {
            stop_word_counter[word] = 0;
        }

        std::fstream file;
        file.open(filepath, std::ios::in);
        std::string word;

        std::unordered_map<std::string, double> wordmap;
        int numWords = 0;
        while (file >> word) {
            std::transform(word.begin(), word.end(), word.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            std::string sanitized;
            std::remove_copy_if(word.begin(), word.end(), std::back_inserter(sanitized), 
                                [](char c) { return std::ispunct(c); });

            if (stop_word_counter.find(sanitized) == stop_word_counter.end() && sanitized.length() > 0) {
                wordmap[stem(sanitized)] += 1;
                numWords += 1;
            }
        }

        // std::for_each(wordmap.begin(), wordmap.end(), 
        //       [numWords](std::pair<const std::string, double>& p) {
        //         p.second = p.second / numWords;
        //       });

        return wordmap;
}



int main() {

    std::unordered_map<std::string, double> wmp = tf("../testset/sci.space/60151");

    using namespace std::chrono;

    long double sum = 0;
    long long int N = 10000;

    auto start_global = high_resolution_clock::now();

    for (long long int i = 0; i < N; i++) {

        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }

        auto start = high_resolution_clock::now();

        std::unordered_map<std::string, double> wordmap = tf("../testset/sci.space/60151");

        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        sum += duration.count();
    }

    auto stop_global = high_resolution_clock::now();

    std::cout << "Average time taken by function: "
         << sum / N << " microseconds" << std::endl;

    auto duration_global = duration_cast<microseconds>(stop_global - start_global);

    std::cout << "Total time taken by function: "
         << duration_global.count() << " microseconds" << std::endl;

    return 0;

    // std::unordered_map<std::string, double> wordmap = tf("../testset/sci.space/60151");

    // for (auto word : wordmap) {
    //     std::cout << word.first << " " << word.second << std::endl;
    // }
}

