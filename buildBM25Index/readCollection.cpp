// g++ -O3 -std=c++20 readCollection.cpp -o readCollection

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string.h>
#include <algorithm>
#include <regex>
#include <filesystem>

const int FILE_PER_DOC = 3907; 
const int NUM_FILES = 256; 

uint64_t pack(uint32_t termID, uint32_t docID);
void tokenizeString(const std::string& line, std::vector<std::string>& tokens);
void writeTempFile(const std::vector<uint64_t>& buffer, int iter);
void writeOtherFiles(const std::vector<std::string>& termToWord,
    const std::unordered_map<uint32_t, size_t>& pageTable);


int main() {
    std::filesystem::create_directory("tempFiles");

    std::ifstream subset("../ms_marco/msmarco_passages_subset.tsv");
    if (!subset) { std::cerr << "Unable to open subset.tsv"; exit(1); }
    std::ifstream collection("../ms_marco/collection.tsv");
    if (!collection) { std::cerr << "Unable to open collection.tsv"; exit(1); }

    std::unordered_map<std::string, uint32_t> lexicon;
    std::vector<std::string> termToWord;
    uint32_t currTermID = 0;

    std::unordered_map<uint32_t, size_t> pageTable;

    std::vector<uint64_t> buffer;
    std::vector<std::string> tokens;

    uint32_t subsetDocID;
    std::vector<uint32_t> subsetList;
    while (subset >> subsetDocID) {
        subsetList.push_back(subsetDocID);
    }
    std::sort(subsetList.begin(), subsetList.end());

    std::cout << subsetList.size() << std::endl;

    std::string line = ""; 
    size_t subsetIndex = 0;
    for (int i = 0; i < NUM_FILES; i++) {
        for (int j = 0; j < FILE_PER_DOC && std::getline(collection, line); j++) {
            uint32_t currSubset = subsetList[subsetIndex++];

            size_t tab = line.find('\t');
            // std::cout << j << std::endl;
            uint32_t docId = static_cast<uint32_t>(std::stoul(line.substr(0, tab)));
            std::string passage = line.substr(tab+1);

            while (docId != currSubset) {
                std::getline(collection, line);
                tab = line.find('\t');
                docId = static_cast<uint32_t>(std::stoul(line.substr(0, tab)));
                passage = line.substr(tab+1);
            }

            tokenizeString(passage, tokens);
            pageTable.insert({docId, tokens.size()}); // update page table with new docid

            for (const std::string& token : tokens) {
                if (lexicon.find(token) == lexicon.end()) { // checks if word already has a termID
                    lexicon.insert({token, currTermID++});
                    termToWord.push_back(token);
                }

                buffer.push_back(pack(lexicon[token], docId));
            }

            if (subsetIndex == subsetList.size()) { break; }
        }

        if (buffer.empty()) { break; } // could be done early

        std::sort(buffer.begin(), buffer.end());
        writeTempFile(buffer, i);
        buffer.clear();
    }
    collection.close();

    std::cout << pageTable.size() << std::endl;

    writeOtherFiles(termToWord, pageTable);
    return 0;
}

/*
Returns a 64 bit unsigned integer in the following representation:
-- 32 bits (termID) -- 32 bits (docID) --
*/
uint64_t pack(uint32_t termID, uint32_t docID) {
    return (uint64_t(termID) << 32) | docID;
}

/*
Splits and normalizes the string into tokens of all lowercase words without
nonalphanumeric characters except those within words
*/
void tokenizeString(const std::string& line, std::vector<std::string>& tokens) {
    tokens.clear();
    std::string tempString;
    
    for (char ch : line) {
        if (isalnum(ch)) { tempString.push_back((char)tolower(ch));}
        else { 
            if (tempString.size() == 0) { continue; }
            tokens.push_back(tempString);
            tempString.clear();
        }
    }
    if (!tempString.empty()){
        tokens.push_back(tempString);
        tempString.clear();
    }

}

/*
Given a vector of packed integers, groups together all the numbers so we get files of
(packedNum, freq) which can be unpacked into (termID, docID, freq). 

TODO: consider impact score instead of freq -> will need size of token vector from earlier
*/
void writeTempFile(const std::vector<uint64_t>& buffer, int iter) {
    std::string fileName = "tempFiles/temp" + std::to_string(iter);
    std::ofstream output(fileName);
    if (!output) { std::cerr << "Failed to open stream for temp file"; exit(1); }

    int currCount = 1;
    uint64_t currNum = buffer[0];
    for (size_t i = 1; i < buffer.size(); i++) {
        if (buffer[i] == currNum)  { currCount += 1; }
        else {
            output << currNum << " " << currCount << " ";
            currNum = buffer[i];
            currCount = 1;
        }
        // output << std::endl;
    }
    output << currNum << " " << currCount << " "; // for final object
    output.close();
}

/*
Writes the lexicon v1, termToWord, and page table to disk
*/
void writeOtherFiles(const std::vector<std::string>& termToWord,
    const std::unordered_map<uint32_t, size_t>& pageTable) {

    std::ofstream termToWordOutput("tempFiles/termToWord");
    if (!termToWordOutput) { std::cerr << "Failed to open stream for termToWord"; exit(1); }
    for (const std::string& entry : termToWord) {
        termToWordOutput << entry << " ";
    }
    termToWordOutput.close();

    std::ofstream pageTableOutput("tempFiles/pageTable");
    if (!pageTableOutput) { std::cerr << "Failed to open stream for pageTable"; exit(1); }
    for (const auto& entry : pageTable) {
        pageTableOutput << entry.first << " " << entry.second << " ";
    }
    pageTableOutput.close();
}
