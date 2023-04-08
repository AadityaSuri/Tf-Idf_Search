#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <unordered_map>

std::vector<std::string> docPreProcessing(std::string filepath);
std::unordered_map<std::string, int> tf(std::vector<std::string> docTextList);