// assume 6400b blocks, lets say 10 chunks per block, chunks will contain 128 entries of docids/freq
// max 32bits per docid, 1byte per freq
// each entry in the mergedPreIndex looks like (packedNum, freq) = (termid + docid, freq)
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string.h>
#include <algorithm>
#include <regex>
#include <filesystem>
#include "ChunkOverhead.h"

// struct used to hold metadata that will be useful for reading in the 
// index later
struct lexiconData {
    uint32_t startByte;
    uint32_t startChunk;
    uint32_t elemsFirstChunk;
    uint32_t elemsLastChunk;
    uint32_t firstChunkPos;
    uint32_t lastChunkPos;
    std::vector<std::pair<uint32_t, uint32_t>> nextBytes_lastDocID;
    // nextBytes is the amount of bytes needed to read with in that chunk
    // lastDocID is that for each chunk
};


void printLex(const std::unordered_map<uint32_t, lexiconData>& lexicon, 
    uint32_t termID, size_t maxEntries);
uint32_t unpackTermID(uint64_t pack);
uint32_t unpackDocID(uint64_t pack);
void arrDifferences(uint32_t* arr, int start, int end);
uint16_t encodeNum(std::ofstream* output, uint32_t num);
void readVector(std::vector<std::string>& words);
void byteWrite(std::ofstream* output, uint32_t num, int size);
void writeLex(std::ofstream& lexFile, std::unordered_map<uint32_t, 
    lexiconData> lexicon);

int main() {
    std::ifstream preind("mergedPreIndex");
    if (!preind) { std::cerr << "Unable to open mergedPreIndex.txt"; exit(1); }
    std::ofstream index("index.txt", std::ios::binary | std::ios::in | std::ios::out);
    std::ofstream lexiconFile("lexicon.txt");
    if (!index.is_open()) {
        std::ofstream temp("index.txt", std::ios::binary);
        temp.close();
        index.open("index.txt", std::ios::binary | std::ios::in | std::ios::out);
    }
    if (!lexiconFile) 
    { std::cerr << "Unable to open an lexicon file\n"; exit(1); }

    int count = 0;
    uint32_t freq;
    uint64_t packedNum;

    
    std::unordered_map<uint32_t, lexiconData> lexicon; 
    uint32_t currChunk = 0;
    ChunkOverhead buffer = ChunkOverhead(&index);
    uint32_t prevTermID = 0;
    uint32_t termid;
    uint32_t docid;
    std::vector<std::pair<uint32_t, uint32_t>> tempSkip_lastDocID; 
    // first is how many bytes are are supposed to read in the curr chunk, 
    // second is the last docid for skipping
    while (true){ 
        preind >> packedNum; // get the packed num
        preind >> freq; // next is the freq
        
        termid = unpackTermID(packedNum);
        docid = unpackDocID(packedNum);
        
        if (count == 0){
            std::vector<std::pair<uint32_t, uint32_t>> nextSkip_lastDocID;
            nextSkip_lastDocID.push_back({0, 0});
            lexicon[termid] = lexiconData{0, 0, 0, 0, 0, 0, nextSkip_lastDocID};
             // set up first entry into the lexicon
        }
    
    
        if (prevTermID != termid){
            lexicon[prevTermID].elemsLastChunk = buffer.currListInd - buffer.flushedListInd;
            lexicon[prevTermID].lastChunkPos = buffer.currListInd-1;
            if (currChunk == lexicon[prevTermID].startChunk){
                lexicon[prevTermID].elemsFirstChunk = buffer.currListInd - buffer.flushedListInd;
            }
            buffer.subtractionCompress();
            uint32_t size = buffer.flushDocIDs();
            lexicon[prevTermID].nextBytes_lastDocID.push_back({size, docid}); // no one knows what the 
            // last docid is yet. I will trust the next termid to fill it in for me once it finds out
            
            lexicon[termid] = lexiconData{static_cast<uint32_t>(index.tellp()),
                                            currChunk,
                                            0,
                                            0, 
                                            0,
                                            buffer.currListInd,
                                            tempSkip_lastDocID};
            
        }
        prevTermID = termid;

        buffer.addToChunk(docid, (uint8_t)freq);

        if (buffer.currListInd == CHUNK_LIST_SIZE){ 
            // check if this is the first flush for the curr lexicon entry
            if (currChunk == lexicon[prevTermID].startChunk){
                lexicon[prevTermID].elemsFirstChunk = buffer.currListInd - buffer.flushedListInd;
            }
            buffer.subtractionCompress();
            uint32_t size = buffer.flush();
            lexicon[prevTermID].nextBytes_lastDocID.push_back({size, docid});
            buffer.reset(); // flushInd = 0, currInd = 0 15595716

            currChunk++;

        }
        count++;
        if (!preind){
            if (currChunk == lexicon[prevTermID].startChunk){
                lexicon[prevTermID].elemsFirstChunk = buffer.currListInd - buffer.flushedListInd;
            }
            lexicon[prevTermID].elemsLastChunk = buffer.currListInd - buffer.flushedListInd;
            lexicon[prevTermID].lastChunkPos = buffer.currListInd-1;
            buffer.subtractionCompress();
            uint32_t size = buffer.flush();
            lexicon[prevTermID].nextBytes_lastDocID.push_back({size, docid});
            break;
        }

    }
    writeLex(lexiconFile, lexicon);

}




/*
Returns the first 32 bits of the packed number, which is the termID 8841823
*/
uint32_t unpackTermID(uint64_t pack) {
    return uint32_t(pack >> 32);
}

/*
Returns the last 32 bits of the packed number, which is the docID
*/
uint32_t unpackDocID(uint64_t pack) {
    return uint32_t(pack & 0xffffffffu);
}


// used to read the termToWord file so we can convert between termid to an 
// actual string word
void readVector(std::vector<std::string>& words) {
    std::ifstream termToWord("tempFiles/termToWord");
    if (!termToWord) { std::cerr << "Failed to open termToWord\n"; exit(1); }

    std::string holder;
    while (termToWord >> holder) { words.push_back(holder); }
    termToWord.close();
}

void writeLex(std::ofstream& lexFile, std::unordered_map<uint32_t, lexiconData> lexicon) {
    std::vector<std::string> termToWord;
    readVector(termToWord); 
    for (const auto& [term, data] : lexicon) {
        lexFile << termToWord[term] << " " << 
                    data.startByte  << " " << 
                    data.elemsFirstChunk << " " << 
                    data.elemsLastChunk << " " << 
                    data.firstChunkPos << " " << 
                    data.lastChunkPos << " ";
        for (const auto& pair : data.nextBytes_lastDocID) {
            lexFile << pair.first << " " << pair.second << " ";
        }
        lexFile << std::endl;
    }
}

void printLex(const std::unordered_map<uint32_t, lexiconData>& lexicon, uint32_t termID, size_t maxEntries = 20) {
    auto it = lexicon.find(termID);
    if (it == lexicon.end()) {
        std::cerr << "Term ID " << termID << " not found in lexicon.\n";
        return;
    }

    const lexiconData& lex = it->second;

    std::cout << "lexicon[" << termID << "] {\n"
              << "  startByte: " << lex.startByte << "\n"
              << "  startChunk: " << lex.startChunk << "\n"
              << "  elemsFirstChunk: " << lex.elemsFirstChunk << "\n"
              << "  elemsLastChunk: " << lex.elemsLastChunk << "\n"
              << "  firstChunkPos: " << lex.firstChunkPos << "\n"
              << "  lastChunkPos: " << lex.lastChunkPos << "\n"
              << "  nextBytes_lastDocID: [\n";

    size_t toPrint = std::min(maxEntries, lex.nextBytes_lastDocID.size());
    for (size_t i = 0; i < toPrint; i++) {
        const auto& p = lex.nextBytes_lastDocID[i];
        std::cout << "    { nextBytes: " << p.first
                  << ", lastDocID: " << p.second << " }\n";
    }

    if (toPrint < lex.nextBytes_lastDocID.size()) {
        std::cout << "    ... (" << lex.nextBytes_lastDocID.size() - toPrint
                  << " more entries)\n";
    }

    std::cout << "  ]\n}\n";

    std::cout << "  ]\n}\n";
}