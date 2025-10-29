#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <string>
#include <filesystem>
#include <thread>
#include <cstdint>

const int NUM_MERGE = 16;

struct queueObject {
    uint64_t packedNum;
    int freq;
    int streamNum;
};

// comparison functor for priority queue to be min-heap
struct ComparePacked {
    bool operator()(const queueObject& a, const queueObject& b) const {
        return a.packedNum > b.packedNum; 
    }
};

void mergeFiles(int startNum, const std::string& path, const std::string& writePath);
void readIntoQueue(std::ifstream& stream, std::queue<queueObject>& queue, int streamNum);

int main() {
    std::filesystem::create_directory("tempFilesMerged");
    std::vector<std::thread> threads;
    threads.reserve(NUM_MERGE);

    for (int i = 0; i < NUM_MERGE; i++) { // merge groups of NUM_MERGE into 1, resulting in NUM_FILES / NUM_MERGE semi-merged files
        threads.emplace_back(mergeFiles, NUM_MERGE * i, "tempFiles/temp", "tempFilesMerged/temp" + std::to_string(i));
    }

    for (int i = 0; i < NUM_MERGE; i++) {
        threads[i].join();
    }

    mergeFiles(0, "tempFilesMerged/temp", "mergedPreIndex"); // merge 16 files into 1
}

void mergeFiles(int startNum, const std::string& path, const std::string& writePath) {
    std::ofstream outputStream(writePath);
    if (!outputStream) { std::cerr << "Failed to open output stream"; exit(1); }

    std::vector<std::ifstream> inputStreams;
    for (int i = startNum; i < startNum+NUM_MERGE; i++) {
        inputStreams.emplace_back(path + std::to_string(i));
        if (!inputStreams.back()) { std::cerr << "Failed to open input stream " << i; exit(1); }
    }

    std::vector<std::queue<queueObject>> inputBuffers(NUM_MERGE);
    std::priority_queue<queueObject, std::vector<queueObject>, ComparePacked> heap;

    for (int i = 0; i < NUM_MERGE; i++) { // get initial state for buffers and move one element to heap
        readIntoQueue(inputStreams[i], inputBuffers[i], i);
        heap.push(inputBuffers[i].front());
        inputBuffers[i].pop();
    }

    while (!heap.empty()) { // when removing an item from the heap, attempt to replace it with the next element from the same buffer
        queueObject currObject = heap.top();
        heap.pop();

        outputStream << currObject.packedNum << " " << currObject.freq << " ";

        int currStream = currObject.streamNum;
        if (inputBuffers[currStream].empty()) { // refill buffer if empty
            readIntoQueue(inputStreams[currStream], inputBuffers[currStream], currStream); 
            if (inputBuffers[currStream].empty()) { continue; } // if no more items, don't push
        }

        heap.push(inputBuffers[currStream].front());
        inputBuffers[currStream].pop();
    }

    outputStream.close();
    for (int i = 0; i < NUM_MERGE; i++) {
        inputStreams[i].close();
    }
}

void readIntoQueue(std::ifstream& stream, std::queue<queueObject>& queue, int streamNum) {
    uint64_t packedNum; int freq;
    for (int i = 0; i < 20 && stream >> packedNum >> freq; i++) {
        queue.push(queueObject{packedNum, freq, streamNum});
    }
}