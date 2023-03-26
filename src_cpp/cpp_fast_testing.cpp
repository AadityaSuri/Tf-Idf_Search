#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iomanip>


std::unordered_map<std::string, double> tf(std::vector<std::string> wordlist) {
    std::unordered_map<std::string, double> wordmap;

    for (auto word : wordlist) {
        wordmap[word] += 1;
    }
    int numWords = wordlist.size();
    for (auto word : wordmap) {
        wordmap[word.first] = wordmap[word.first] / numWords;
    }
    return wordmap;
} 

std::unordered_map<std::string, double> tf(std::string filepath) {
    std::vector<std::string> stop_words = {"i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't"};

        std::unordered_map<std::string, int> stop_word_counter;
        for (auto word : stop_words) {
            stop_word_counter[word] = 0;
        }

        std::fstream file;
        file.open(filepath, std::ios::in);
        std::string word;

        std::unordered_map<std::string, double> wordmap;
        int numWords = 0;
        while (file >> word) {
            if (stop_word_counter.find(word) == stop_word_counter.end()) {
                std::string punc_removed;
                std::remove_copy_if(word.begin(), word.end(), std::back_inserter(punc_removed), 
                                    [](char c) { return std::ispunct(c); });
                numWords++;
                wordmap[punc_removed] += 1;
            }
        }

        for (auto word : wordmap) {
            wordmap[word.first] = wordmap[word.first] / numWords;
        }

        return wordmap;
}

int main() {

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
}


