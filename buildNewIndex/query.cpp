#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <queue>
#include <sstream>
#include <cstring>

const double K1 = 1.2;
const double B = 0.75;
const int N = 1000000;
const double DAVG = 57.7154;
const uint8_t CHUNK_SIZE = 128;

struct Chunk {
    std::vector<uint8_t> compressedDocIds;
    uint8_t freq[CHUNK_SIZE];
}; 

struct UncompressedChunk {
    uint32_t docIds[CHUNK_SIZE];
    uint8_t freq[CHUNK_SIZE];
    uint8_t currPos = 0;
    UncompressedChunk(){
        memset(docIds, 0, sizeof(docIds));
        memset(freq, 0, sizeof(freq));
    }
};

struct InvertedList {
    std::vector<uint32_t> lastDocIds;
    std::vector<uint32_t> docIdBytes;
    std::vector<Chunk*> compressedChunks;
    uint32_t numDocs;
    uint32_t currChunk = 0;
    UncompressedChunk* currUncompressedChunk = nullptr;
    uint8_t elemsInFirstChunk;
    uint8_t elemsInLastChunk;
   
};

struct LexiconInvertedList {
    std::vector<uint32_t> docIdBytes;   
    std::vector<uint32_t> lastDocIds;   
    int32_t startByte = 0;
    uint32_t elemsFirstChunk = 0;
    uint32_t elemsLastChunk = 0;
    uint32_t firstChunkPos = 0;
    uint32_t lastChunkPos = 0;
};

struct Compare {
    bool operator()(const std::pair<double, uint32_t>& a,
                    const std::pair<double, uint32_t>& b) const {
        return a.first > b.first;  // min-heap based on the double
    }
};

uint32_t decodeNumFromFile(std::ifstream& file);
void readPageTable(std::unordered_map<uint32_t, uint16_t>&);
void readLexicon(std::unordered_map<std::string, LexiconInvertedList*>&);
void tokenizeString(const std::string& line, std::vector<std::string>& tokens);
double bm25(InvertedList* currList, uint16_t docLen);
uint32_t findNextDocID(InvertedList* currList, uint32_t target);
InvertedList* openInvertedList(LexiconInvertedList*, std::ifstream&);
void conjunctiveDAAT(std::vector<std::pair<uint32_t, InvertedList*>>& lists, 
    const std::unordered_map<uint32_t, uint16_t>& pageTable);
void disjunctiveDAAT(std::vector<std::pair<uint32_t, InvertedList*>>& lists, 
    const std::unordered_map<uint32_t, uint16_t>& pageTable);

int main() {
    std::ifstream index("index.txt", std::ios::binary);
    if (!index) {std::cout << "cant open index" << std::endl; exit(1);}
    std::unordered_map<uint32_t, uint16_t> pageTable;
    readPageTable(pageTable);
    std::unordered_map<std::string, LexiconInvertedList*> lexicon;
    readLexicon(lexicon);
    std::string query; bool mode; std::vector<std::string> tokens;

    while (true) {
        std::cout << "Enter query: ";
        std::getline(std::cin, query);
        std::cout << "Enter 0 for conjunctive and 1 for disjunctive: ";
        std::cin >> mode;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        tokenizeString(query, tokens);
        std::vector<std::pair<uint32_t, InvertedList*>> lists;

        for (const std::string& token : tokens) { 
            auto it = lexicon.find(token);
            if (it == lexicon.end()) {
                std::cerr << "[WARN] docID " << token << " missing from lexicon\n";
                continue;
            }
            InvertedList* currList = openInvertedList(lexicon[token], index);
            lists.push_back(std::pair<uint32_t, InvertedList*>(currList->numDocs, currList));
        }

        std::sort(lists.begin(), lists.end());

        if (!mode) { conjunctiveDAAT(lists, pageTable); }
        else { disjunctiveDAAT(lists, pageTable); }

        for (size_t i = 0; i < lists.size(); i++) {
            InvertedList* currList = lists[i].second;
            for (Chunk* chunk : currList->compressedChunks) {
                delete chunk;
            }
            delete currList;
        }
    }
    return 0;
}

/*
Reads necessary bytes following the varbyte decoding to decode one number
from the input stream
*/
uint32_t decodeNumFromFile(std::ifstream& file) {
    uint32_t num = 0;
    uint32_t shift = 0;
    uint8_t currByte = 0;

    // Keep reading bytes until a byte < 128 is found
    while (true) {
        if (!file.read(reinterpret_cast<char*>(&currByte), sizeof(uint8_t))) {
            throw std::runtime_error("Unexpected EOF while decoding number");
        }

        if (currByte >= 128) { // first bit is 1
            num |= (currByte & 127) << shift;
            shift += 7;
        } else {
            num |= currByte << shift;
            break;
        }
    }

    return num;
}

/*
Reads necessary bytes following the varbyte decoding to decode one number
from a vector of bytes
*/
uint32_t decodeNum(const std::vector<uint8_t>& bytes, size_t& currPos) {
    uint32_t num = 0;
    uint32_t shift = 0;
    uint8_t currByte;

    while ((currByte = static_cast<uint8_t>(bytes[currPos++])) >= 128) { // first bit is 1
        num = num + ((currByte & 127) << shift);
        shift += 7;
    }

    return num + (currByte << shift);
}

/*
Reads from page table file, which contains pairs of docID, docSize which is 
stored into a map for processing in BM25
*/
void readPageTable(std::unordered_map<uint32_t, uint16_t>& pageTable) {
    std::ifstream pageTableFile("tempFiles/pageTable");
    if (!pageTableFile) { std::cerr << "Unable to open page table file\n"; exit(1); }

    uint32_t tempDocId; uint16_t tempDocSize;
    uint32_t totalDocSize = 0;
    while (pageTableFile >> tempDocId >> tempDocSize) {
        pageTable.insert({tempDocId, tempDocSize});
        totalDocSize += tempDocSize;
    }

    std::cout << "Document AVG Len: " << totalDocSize / (float)pageTable.size() << std::endl;

    pageTableFile.close();
}

/*
Reads from lexicon file, which contains the following metadata:
Word, startByte, elements in first chunk, elements in last chunk, first chunk position, last chunk position
Then loops through reading pairs of docIDBytes, lastDocID until the next term is reached.

Stored within lexicon map, which maps the word to the struct of metadata from the lexicon
*/
void readLexicon(std::unordered_map<std::string, LexiconInvertedList*>& lexicon) {
    std::ifstream lexiconStream("lexicon.txt");
    if (!lexiconStream) { 
        std::cerr << "Unable to open lexicon\n"; 
        exit(1); 
    }

    std::string line;
    while (std::getline(lexiconStream, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);

        std::string word;
        uint32_t startByte, elemsFirstChunk, elemsLastChunk, firstChunkPos, lastChunkPos;

        ss >> word >> startByte >> elemsFirstChunk >> elemsLastChunk >> firstChunkPos >> lastChunkPos;

        LexiconInvertedList* currLexiconEntry = new LexiconInvertedList{};
        currLexiconEntry->startByte = startByte;
        currLexiconEntry->elemsFirstChunk = elemsFirstChunk;
        currLexiconEntry->elemsLastChunk = elemsLastChunk;
        currLexiconEntry->firstChunkPos = firstChunkPos;
        currLexiconEntry->lastChunkPos = lastChunkPos;

        // Read remaining numbers as nextBytes_lastDocID pairs
        uint32_t nextBytes;
        int lastDocID;
        while (ss >> nextBytes >> lastDocID) {
            currLexiconEntry->docIdBytes.push_back(nextBytes);
            currLexiconEntry->lastDocIds.push_back(lastDocID);
        }

        lexicon[word] = currLexiconEntry;
    }
}

/*
Splits and normalizes the string into tokens of all lowercase words without
nonalphanumeric characters except those within words

*/
void tokenizeString(const std::string& line, std::vector<std::string>& tokens) {
    static const std::unordered_set<std::string> stopWords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "has", "he", "in", "is", "it", "its", "of", "on",
        "that", "the", "to", "was", "were", "will", "with", "this",
        "these", "those", "their", "they", "i", "you", "your",
        "she", "his", "her", "them", "or", "but", "not", "we",
        "what", "which", "who", "when", "where", "why", "how"
    };

    tokens.clear();
    std::string tempString;
    
    for (char ch : line) {
        if (isalnum(ch)) {
            tempString.push_back((char)tolower(ch));
        } else {
            if (tempString.empty()) continue;
            if (!stopWords.count(tempString)) {
                tokens.push_back(tempString);
            }
            tempString.clear();
        }
    }

    if (!tempString.empty() && !stopWords.count(tempString)) {
        tokens.push_back(tempString);
    }

    // Remove duplicates
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
}

/*
Calculates BM25 score for a single term in a single document
*/
double bm25(InvertedList* currList, uint16_t docLen) {
    uint32_t ft = currList->numDocs;
    uint8_t fdt = currList->currUncompressedChunk->freq[currList->currUncompressedChunk->currPos];
    double K = K1 * ((1-B) + B * (docLen) / DAVG);
    return std::log2((N - ft + 0.5) / (ft + 0.5)) * ((K1 + 1) * fdt) / (K + fdt);
}

/*
Determines the next docID in the inverted list that is greater than or equal to the target
*/
uint32_t findNextDocID(InvertedList* currList, uint32_t target) {
    uint32_t currChunk = currList->currChunk;
    // while valid, check lastDocIds to skip chunks where all docIDs are smaller than target
    while (currChunk < currList->lastDocIds.size() && target > currList->lastDocIds[currChunk]) { currChunk++; }
    // if outside of bounds, return number of documents
    if (currChunk >= currList->lastDocIds.size()) { return N; }

    // if not the same chunk, delete old uncompressed chunk and uncompress current chunk
    if (currChunk != currList->currChunk || !currList->currUncompressedChunk) { 
        delete currList->currUncompressedChunk;

        currList->currUncompressedChunk = new UncompressedChunk{};
        currList->currChunk = currChunk;
        Chunk* newChunk = currList->compressedChunks[currChunk];
        
        int chunkLen;
        if (currChunk == 0)
            chunkLen = currList->elemsInFirstChunk;
        else if (currChunk + 1 == currList->lastDocIds.size())
            chunkLen = currList->elemsInLastChunk;
        else
            chunkLen = CHUNK_SIZE;

        size_t index = 0;
        for (int i = 0; i < chunkLen; i++) { // decode varByte
            currList->currUncompressedChunk->docIds[i] = decodeNum(newChunk->compressedDocIds, index);
            currList->currUncompressedChunk->freq[i] = newChunk->freq[i];
        }
        for (int i = 1; i < chunkLen; i++) { // iteratively sum to get actual docIDs
            currList->currUncompressedChunk->docIds[i] += currList->currUncompressedChunk->docIds[i - 1];
        }
        
    }

    int chunkLen;
    if (currChunk == 0)
        chunkLen = currList->elemsInFirstChunk;
    else if (currChunk + 1 == currList->lastDocIds.size())
        chunkLen = currList->elemsInLastChunk;
    else
        chunkLen = CHUNK_SIZE;
    for (int i = 0; i < chunkLen; i++) {  // loop through chunk to find docID >= target
        if (currList->currUncompressedChunk->docIds[i] >= target) { 
            currList->currUncompressedChunk->currPos = i;
            return currList->currUncompressedChunk->docIds[i]; 
        }
    }

    return N; // if somehow not found, return number of documents
}

/*
Helper function to return the elements in the chunk dependent on position
*/
static inline uint32_t elemsInChunk(const LexiconInvertedList* e, uint32_t i) {
    if (i == 0) return e->elemsFirstChunk;
    if (i + 1 == e->docIdBytes.size()) return e->elemsLastChunk;
    return CHUNK_SIZE;
}

/*
Helper function to return the number of postings from an inverted list
*/
static inline uint32_t totalPostings(const LexiconInvertedList* e) {
    if (e->docIdBytes.size() == 1) return e->elemsFirstChunk;
    return e->elemsFirstChunk + (e->docIdBytes.size() - 2) * CHUNK_SIZE + e->elemsLastChunk;
}

/*
Uses decodeNumFromFile to skip a number of elements and place stream at beginning
of valid readable IDs
*/
void skipElems(std::ifstream& index, uint32_t numElems){
    for (uint32_t i=0;i<numElems;i++){
        decodeNumFromFile(index);
    }
}

/*
Opens the inverted list for a given term
*/
InvertedList* openInvertedList(LexiconInvertedList* lexiconMetadata, std::ifstream& index) {
    InvertedList* currList = new InvertedList{};

    currList->docIdBytes = lexiconMetadata->docIdBytes;
    currList->lastDocIds = lexiconMetadata->lastDocIds;
    currList->elemsInFirstChunk = lexiconMetadata->elemsFirstChunk;
    currList->elemsInLastChunk = lexiconMetadata->elemsLastChunk;
    currList->numDocs = totalPostings(lexiconMetadata);

    index.seekg(static_cast<std::streamoff>(lexiconMetadata->startByte), std::ios::beg); // go to beginning of inverted list in index file
    for (uint32_t i=0;i<lexiconMetadata->docIdBytes.size();i++){
        Chunk* chunk = new Chunk{};
        chunk->compressedDocIds.resize(currList->docIdBytes[i]); // resize vector to copy bytes over into underlying array
        index.read(reinterpret_cast<char*>(chunk->compressedDocIds.data()), static_cast<std::streamsize>(currList->docIdBytes[i])); // read in docID bytes
        if (i==0){ // if first chunk, skip the starting docID bytes and frequency bytes that corresponded to a previous term
            skipElems(index, CHUNK_SIZE-(lexiconMetadata->elemsFirstChunk + lexiconMetadata->firstChunkPos));
            index.seekg(static_cast<std::streamoff>(index.tellg()) + lexiconMetadata->firstChunkPos, std::ios::beg);
        }
        else if (i == lexiconMetadata->docIdBytes.size()-1){ // if last chunk, skip remaining docID bytes to get frequency bytes
            skipElems(index, CHUNK_SIZE-(lexiconMetadata->elemsLastChunk));

        }
        index.read(reinterpret_cast<char*>(chunk->freq), static_cast<std::streamsize>(elemsInChunk(lexiconMetadata, i)));

        
        currList->compressedChunks.push_back(chunk);
    }

    return currList;
}

/*
Computes top 10 queries using conjunctiveDAAT processing
*/
void conjunctiveDAAT(std::vector<std::pair<uint32_t, InvertedList*>>& lists, 
    const std::unordered_map<uint32_t, uint16_t>& pageTable) {

    std::cout << "Doing conjunctive DAAT" << std::endl;
    InvertedList* baseList = lists[0].second; // obtain the smallest list to loop over
    std::cout << "Base List contains " << baseList->numDocs << " documents.\n";
    std::priority_queue<std::pair<double, uint32_t>, std::vector<std::pair<double, uint32_t>>, Compare> heap;

    uint32_t currDocId = 0;
    while ((currDocId = findNextDocID(baseList, currDocId)) != N) {
        size_t index = 1;
        uint32_t res = 0;
        for (; index < lists.size(); index++) { // attempt to find docID in all remaining lists
            res = findNextDocID(lists[index].second, currDocId);
            if (res != currDocId || res == N ) { break; }
        }
        if (res == N) break; // if a list returns N, there is nothing else to process
        
        bool missing = false;
        if (index == lists.size()) { // calculate impact score for all lists
            double impactScore = 0;
            for (size_t j = 0; j < lists.size(); j++) {
                    InvertedList* currList = lists[j].second;
                    auto it = pageTable.find(currDocId);
                    if (it == pageTable.end()) {
                        std::cerr << "[WARN] docID " << currDocId << " missing from pageTable\n";
                        missing = true;
                        break;
                    }
                    impactScore += bm25(currList, it->second);
            }

            if (!missing) { // add to heap if possible, instant add if less than 10 and replacing smallest elem if at 10 elements
                if (heap.size() != 10) { heap.push(std::pair<double, uint32_t>(impactScore, currDocId)); }
                else if (heap.top().first < impactScore) {
                    heap.pop();
                    heap.push(std::pair<double, uint32_t>(impactScore, currDocId));  
                }
            }
        }

        currDocId++;
    }

    std::vector<std::pair<double, uint32_t>> topSearches;
    while (!heap.empty()) {
        topSearches.push_back(heap.top());
        heap.pop();
    }
    
    for (size_t i = topSearches.size(); i > 0; i--) {
        std::cout << "Impact Score: " << topSearches[i-1].first << " DocID: " << topSearches[i-1].second << std::endl;
    }
}

/*
Computes top 10 queries using disjunctiveDAAT processing
*/
void disjunctiveDAAT(std::vector<std::pair<uint32_t, InvertedList*>>& lists, 
    const std::unordered_map<uint32_t, uint16_t>& pageTable) {

    std::cout << "Doing disjunctive DAAT" << std::endl;
    std::priority_queue<std::pair<double, uint32_t>, std::vector<std::pair<double, uint32_t>>, Compare> heap;

    const size_t numEssential = std::max<size_t>(size_t(lists.size() * 0.3), 1); // choose lower third of all lists to be essential

    std::vector<std::pair<uint32_t, double>> essentialDocIds;
    for (size_t i = 0; i < numEssential; i++) { // add all docIDs for those lists into one vector, along with each impact score
        InvertedList* currList = lists[i].second;
        uint32_t currDocId = 0;
        for (size_t currDocIndex = 0; currDocIndex < currList->numDocs; currDocIndex++) {
            currDocId = findNextDocID(currList, currDocId);
            if (currDocId == N) break;
            essentialDocIds.push_back(std::pair<uint32_t, double>(currDocId, bm25(currList, pageTable.at(currDocId))));
            currDocId++;
            
        }
    }

    std::sort(essentialDocIds.begin(), essentialDocIds.end()); // groups by docIDs since pair compares on .first
    std::vector<std::pair<uint32_t, double>> essentialDocIdsNoDup;

    for (size_t i = 1; i < essentialDocIds.size(); i++) { // add docIDs together for similar docIDs
        if (essentialDocIds[i-1].first == essentialDocIds[i].first) {
            essentialDocIds[i].second += essentialDocIds[i-1].second;
        }
        else {
            essentialDocIdsNoDup.push_back(essentialDocIds[i-1]);
        }
    }
    essentialDocIdsNoDup.push_back(essentialDocIds[essentialDocIds.size() - 1]);

    for (size_t i = 0; i < essentialDocIdsNoDup.size(); i++) { // do lookups into remaining lists with all docIDs
        uint32_t currDocId = essentialDocIdsNoDup[i].first;
        double currImpact = essentialDocIdsNoDup[i].second;

        for (size_t j = numEssential; j < lists.size(); j++) { // remaining lists begin at index numEssential, essentials are (0, numEssential - 1)
            if (findNextDocID(lists[j].second, currDocId) == currDocId) {
                if (currDocId == N) break;
                currImpact += bm25(lists[j].second, pageTable.at(currDocId));
            }
        }

        // add to heap if possible, instant add if less than 10 and replacing smallest elem if at 10 elements
        if (heap.size() != 10) { heap.push(std::pair<double, uint32_t>(currImpact, currDocId)); }
        else {
            std::pair<double, uint32_t> minImpact = heap.top();
            if (minImpact.first < currImpact) {
                heap.pop();
                heap.push(std::pair<double, uint32_t>(currImpact, currDocId)); 
            }
        }
    }

    std::vector<std::pair<double, uint32_t>> topSearches;
    while (!heap.empty()) {
        topSearches.push_back(heap.top());
        heap.pop();
    }
    
    for (size_t i = topSearches.size(); i > 0; i--) {
        std::cout << "Impact Score: " << topSearches[i-1].first << " DocID: " << topSearches[i-1].second << std::endl;
    }
}