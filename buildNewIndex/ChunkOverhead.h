#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string.h>
#include <algorithm>
#include <regex>
#include <filesystem>

const int CHUNK_LIST_SIZE = 128;

class Chunk{
public:
    uint32_t docIDList[CHUNK_LIST_SIZE];
    uint8_t freqList[CHUNK_LIST_SIZE];
    Chunk();
    void reset();
};

class ChunkOverhead {
    public:
    uint8_t flushedListInd;
    uint8_t currListInd; // which ind we are in the list of each chunk
    std::ofstream* indexFile;
    Chunk currChunk;

    ChunkOverhead(std::ofstream* indexFile);
        
    void addToChunk(uint32_t newID, uint8_t newFreq);
    uint32_t flush(); 
    void subtractionCompress();
    uint32_t flushDocIDs();
    void reset();
};