#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <chrono>


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