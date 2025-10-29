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
using namespace std;

uint32_t arrDifferences(uint32_t* arr, int start, int end);
uint32_t encodeNum(std::ofstream* output, uint32_t num);
void byteWrite(std::ofstream* output, uint32_t num, int size);
Chunk::Chunk(){
    reset();
}
void Chunk::reset(){
    memset(docIDList, 0, sizeof(docIDList));
    memset(freqList, 0, sizeof(freqList));
}


ChunkOverhead::ChunkOverhead(std::ofstream* indexFile){
    this->indexFile = indexFile;
    reset();
}

    
void ChunkOverhead::addToChunk(uint32_t newID, uint8_t newFreq){
    currChunk.docIDList[currListInd] = newID; // append to docid list
    currChunk.freqList[currListInd] = newFreq; // append to freq list 
    currListInd++;
}


// flushes just docids to the file. This is used so we can get that 
// consistent 128 docids, 128 freq... design in the index file
// This will output how many bytes were taken up after compression
uint32_t ChunkOverhead::flushDocIDs(){
    uint32_t chunkBytes = 0;
    // Write docIDs only
    for (int j = flushedListInd; j < currListInd; j++) {
        chunkBytes += encodeNum(indexFile, currChunk.docIDList[j]);
    }
    flushedListInd = currListInd; // curr in is now the last time we flushed
    return chunkBytes;
}
// Will flush the docids from the last flushInd. Will always flush out all the
// freq
uint32_t ChunkOverhead::flush(){
    uint32_t chunkBytes = 0;
    // Write docIDs
    for (int j = flushedListInd; j < CHUNK_LIST_SIZE && currChunk.freqList[j] !=0; j++) {
        chunkBytes += encodeNum(indexFile, currChunk.docIDList[j]);
    }
    // write freq
    for (int j = 0; j < CHUNK_LIST_SIZE && currChunk.freqList[j] !=0; j++) {
        byteWrite(indexFile, currChunk.freqList[j], sizeof(uint8_t));
    }
    flushedListInd = currListInd;
    return chunkBytes;

} 

// we only compress between the last flush ind and the curr ind. This is 
// to account for inverted Indexes in the middle of chunks. We dont want to 
// reduce the start docid of the start of that inverted index
void ChunkOverhead::subtractionCompress(){
    arrDifferences(currChunk.docIDList, flushedListInd, currListInd-1);
}
void ChunkOverhead::reset(){
    currListInd = 0;
    flushedListInd = 0;
    currChunk.reset();
}



uint32_t arrDifferences(uint32_t* arr, int start, int end){
    uint32_t endVal = arr[end];
    for (int i=end; i>start; i--){
        arr[i] = arr[i] - arr[i-1];
    }
    return endVal;
}

void byteWrite(std::ofstream* output, uint32_t num, int size){
    output->write(reinterpret_cast<const char*>(&num), size);
}

/*
Writes a number compressed using varbyte as bytes into an output stream

If a number is greater than 127, we write the upper bytes with a 1 in the first bit to indicate
the number continues into the next byte, and write the remaining 7 bits of that first byte.

At the end its guaranteed to fit within a 7 bit number
*/

uint32_t encodeNum(std::ofstream* output, uint32_t num) {
    uint32_t count = 1;
    while (num >= 128) {
        uint8_t currByte = 128 + (num & 127);
        output->write(reinterpret_cast<const char*>(&currByte), sizeof(uint8_t));
        num = num >> 7;
        count++;
    }

    uint8_t last = static_cast<uint8_t>(num);
    output->write(reinterpret_cast<const char*>(&last), sizeof(uint8_t));
    return count;
}

