#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>


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

std::vector<std::string> docPreProcessing(std::string filepath) {
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

        std::string word;

        while (file >> word) {
            
        }
}

// def docPreProcessing(filepath):
//     stopwords_dict = Counter(stopwords.words('english'))

//     file = open(filepath, 'r', errors='replace')
//     lines = file.readlines()    
    
//     doctext = ""
//     for line in lines:
//         line = line.translate(str.maketrans('', '', string.punctuation)).strip().lower()
//         line = ' '.join([word for word in line.split() if word not in stopwords_dict])
//         doctext += line
        
//     doctextList = doctext.split()
//     return doctextList



int main() {
    std::vector<std::string> wordlist = {"henryzootorontoedu",
    "henry",
    "spencersubject",
    "hlv",
    "fred",
    "prefab",
    "space",
    "stationarticle",
    "c5133agzxnewscsouiucedu",
    "jbh55289uxacsouiucedu",
    "josh",
    "hopkins",
    "writestitan",
    "iv",
    "launches",
    "aint",
    "cheapgranted",
    "thats",
    "titan",
    "ivs",
    "bought",
    "governemnt",
    "titaniii",
    "actually",
    "cheapest",
    "way",
    "put",
    "pound",
    "space",
    "us",
    "expendablelauncherscase",
    "rather",
    "ironic",
    "poorly",
    "commercialmarket",
    "single",
    "titan",
    "iii",
    "orderproblem",
    "commercial",
    "titan",
    "mm",
    "made",
    "little",
    "attemptmarket",
    "theyre",
    "basically",
    "happy",
    "government",
    "businessdont",
    "want",
    "learn",
    "sell",
    "commerciallysecondary",
    "problem",
    "bit",
    "big",
    "theyd",
    "need",
    "gomultisatellite",
    "launches",
    "la",
    "ariane",
    "complicates",
    "marketingtask",
    "quite",
    "significantlyalso",
    "problems",
    "launch",
    "facilities",
    "wrong",
    "timeget",
    "started",
    "properly",
    "memory",
    "serves",
    "pad",
    "used",
    "marsobserver",
    "launch",
    "come",
    "heavy",
    "refurbishment",
    "workprevented",
    "launches",
    "yearct",
    "launches",
    "mars",
    "observer",
    "onestranded",
    "intelsat",
    "least",
    "one",
    "brothers",
    "reachedorbit",
    "properlywork",
    "one",
    "mans",
    "work",
    "henry",
    "spencer",
    "u",
    "toronto",
    "zoologykipling",
    "henryzootorontoedu",
    "utzoohenry"
    };
    
    using namespace std::chrono;

    auto durationList = std::vector<double>();

    for (int i = 0; i < 100; i++) {
        auto start = high_resolution_clock::now();

        std::unordered_map<std::string, double> wordmap = tf(wordlist);

        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        durationList.push_back(duration.count());
    }

    double sum = 0;
    for (auto duration : durationList) {
        sum += duration;
    }

    std::cout << "Average time taken by function: "
         << sum / durationList.size() << " microseconds" << std::endl;

};