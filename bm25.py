#!/usr/bin/env python3
import sys
import os
import time
import heapq
from typing import List, Tuple, Dict

K1 = 1.2
B = 0.75
N = 8841823
DAVG = 55.9158
CHUNK_SIZE = 128
TOPK_HEAP_SIZE = 10


class Chunk:
    def __init__(self):
        self.compressedDocIds = bytearray()
        self.freq = bytearray(CHUNK_SIZE)


class UncompressedChunk:
    def __init__(self):
        self.docids = [0] * CHUNK_SIZE
        self.freq = [0] * CHUNK_SIZE
        self.curr_pos = 0


class InvertedList:
    def __init__(self):
        self.compressedChunks: List[Chunk] = []
        self.lastDocIds: List[int] = []
        self.docIdBytes: List[int] = []
        self.numDocs: int = 0
        self.currChunk: int = 0
        self.currUncompressedChunk: UncompressedChunk = None
        self.elemsInFirstChunk: int = 0
        self.elemsInLastChunk: int = 0


class LexiconInvertedList:
    def __init__(self):
        self.startByte: int = 0
        self.elemsFirstChunk: int = 0
        self.elemsLastChunk: int = 0
        self.firstChunkPos: int = 0
        self.lastChunkPos: int = 0
        self.docIdBytes: List[int] = []
        self.lastDocIds: List[int] = []


# --- varbyte decode helpers -------------------------------------------------
def decode_varbyte(bytes_seq, start_pos):
    n = 0
    shift = 0
    pos = start_pos
    while True:
        b = bytes_seq[pos]
        pos += 1
        if b >= 128:
            n |= (b & 127) << shift
            shift += 7
        else:
            n |= b << shift
            break
    return n, pos


def decode_num_from_file(f) -> int:
    num = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("decode_num_from_file: unexpected EOF")
        curr_byte = b[0]
        if curr_byte >= 128:
            num |= (curr_byte & 127) << shift
            shift += 7
        else:
            num |= curr_byte << shift
            break
    return num


# --- I/O readers ------------------------------------------------------------
def read_page_table(path="pageTable.txt") -> Dict[int, int]:
    page_table: Dict[int, int] = {}
    if not os.path.exists(path):
        print(f"[WARN] page table file not found: {path}", file=sys.stderr)
        return page_table

    total_length = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            docid = int(parts[0])
            doclen = int(parts[1])
            page_table[docid] = doclen
            total_length += doclen

    if page_table:
        print(f"Number of documents: {len(page_table)}")
        print(f"Average document length: {total_length / len(page_table):.6f}")
    return page_table


def read_lexicon(path="lexicon.txt") -> Dict[str, LexiconInvertedList]:
    lexicon: Dict[str, LexiconInvertedList] = {}
    if not os.path.exists(path):
        print(f"Unable to open lexicon: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            word = parts[0]
            entry = LexiconInvertedList()
            entry.startByte = int(parts[1])
            entry.elemsFirstChunk = int(parts[2])
            entry.elemsLastChunk = int(parts[3])
            entry.firstChunkPos = int(parts[4])
            entry.lastChunkPos = int(parts[5])

            idx = 6
            while idx + 1 < len(parts):
                entry.docIdBytes.append(int(parts[idx]))
                entry.lastDocIds.append(int(parts[idx + 1]))
                idx += 2

            lexicon[word] = entry
    return lexicon


# --- utility functions replicated from C++ ---------------------------------
def elems_in_chunk(lex: LexiconInvertedList, i: int) -> int:
    if i == 0:
        return lex.elemsFirstChunk
    if i + 1 == len(lex.docIdBytes):
        return lex.elemsLastChunk
    return CHUNK_SIZE


def total_postings(lex: LexiconInvertedList) -> int:
    if len(lex.docIdBytes) == 1:
        return lex.elemsFirstChunk
    return lex.elemsFirstChunk + (len(lex.docIdBytes) - 2) * CHUNK_SIZE + lex.elemsLastChunk


def skip_elems_file(f, num_elems: int):
    for _ in range(num_elems):
        _ = decode_num_from_file(f)


def open_inverted_list(lex_entry: LexiconInvertedList, index_file) -> InvertedList:
    inv_list = InvertedList()
    inv_list.docIdBytes = lex_entry.docIdBytes.copy()
    inv_list.lastDocIds = lex_entry.lastDocIds.copy()
    inv_list.elemsInFirstChunk = lex_entry.elemsFirstChunk
    inv_list.elemsInLastChunk = lex_entry.elemsLastChunk
    inv_list.numDocs = sum(
        [lex_entry.elemsFirstChunk] +
        [(CHUNK_SIZE if i != len(lex_entry.docIdBytes) - 1 else lex_entry.elemsLastChunk)
         for i in range(1, len(lex_entry.docIdBytes))]
    )

    index_file.seek(lex_entry.startByte)
    for i, num_bytes in enumerate(lex_entry.docIdBytes):
        chunk = Chunk()
        chunk.compressedDocIds = bytearray(index_file.read(num_bytes))
        chunk_len = CHUNK_SIZE
        if i == 0:
            chunk_len = lex_entry.elemsFirstChunk
        elif i == len(lex_entry.docIdBytes) - 1:
            chunk_len = lex_entry.elemsLastChunk
        chunk.freq = bytearray(index_file.read(chunk_len))
        inv_list.compressedChunks.append(chunk)
    return inv_list


def uncompress_chunk_into(L: InvertedList, chunk_index: int):
    L.currUncompressedChunk = UncompressedChunk()
    L.currChunk = chunk_index
    chunk = L.compressedChunks[chunk_index]

    if chunk_index == 0:
        chunk_len = L.elemsInFirstChunk
    elif chunk_index + 1 == len(L.lastDocIds):
        chunk_len = L.elemsInLastChunk
    else:
        chunk_len = CHUNK_SIZE

    pos_ref = [0]
    docids = []
    for i in range(chunk_len):
        gap, pos_ref[0] = decode_varbyte(chunk.compressedDocIds, pos_ref[0])
        absid = gap if i == 0 else docids[-1] + gap
        docids.append(absid)

    for i in range(chunk_len):
        L.currUncompressedChunk.docids[i] = docids[i]
        L.currUncompressedChunk.freq[i] = chunk.freq[i]

    L.currUncompressedChunk.curr_pos = 0


def find_next_docid(L: InvertedList, target: int) -> int:
    curr_chunk = L.currChunk
    while curr_chunk < len(L.lastDocIds) and target > L.lastDocIds[curr_chunk]:
        curr_chunk += 1
    if curr_chunk >= len(L.lastDocIds):
        return N
    if L.currUncompressedChunk is None or curr_chunk != L.currChunk:
        uncompress_chunk_into(L, curr_chunk)

    if curr_chunk == 0:
        chunk_len = L.elemsInFirstChunk
    elif curr_chunk + 1 == len(L.lastDocIds):
        chunk_len = L.elemsInLastChunk
    else:
        chunk_len = CHUNK_SIZE

    for i in range(chunk_len):
        if L.currUncompressedChunk.docids[i] >= target:
            L.currUncompressedChunk.curr_pos = i
            return L.currUncompressedChunk.docids[i]

    L.currChunk = curr_chunk + 1
    L.currUncompressedChunk = None
    return find_next_docid(L, target)


def bm25(L: InvertedList, doc_len: int) -> float:
    ft = L.numDocs
    if L.currUncompressedChunk is None:
        fdt = 0
    else:
        p = L.currUncompressedChunk.curr_pos
        fdt = L.currUncompressedChunk.freq[p]
    K = K1 * ((1 - B) + B * (doc_len) / DAVG)
    denom = K + fdt
    if denom == 0:
        return 0.0
    import math
    return math.log2((N - ft + 0.5) / (ft + 0.5)) * ((K1 + 1) * fdt) / denom


# --- tokenization ----------------------------------------------------------
STOP_WORDS = {"the"}


def tokenize_string(line: str) -> List[str]:
    tokens: List[str] = []
    temp = []
    for ch in line:
        if ch.isalnum():
            temp.append(ch.lower())
        else:
            if temp:
                word = "".join(temp)
                if word not in STOP_WORDS:
                    tokens.append(word)
                temp = []
    if temp:
        word = "".join(temp)
        if word not in STOP_WORDS:
            tokens.append(word)
    tokens.sort()
    out = []
    prev = None
    for t in tokens:
        if t != prev:
            out.append(t)
        prev = t
    return out


# --- main ------------------------------------------------------------------
def main():
    index_path = "index.txt"
    if not os.path.exists(index_path):
        print("cant open index", file=sys.stderr)
        sys.exit(1)

    page_table = read_page_table("pageTable.txt")
    lexicon = read_lexicon("lexicon.txt")

    with open(index_path, "rb") as index_file:
        while True:
            try:
                query = input("Enter query: ")
            except EOFError:
                break
            mode_in = input("Enter 0 for conjunctive and 1 for disjunctive: ").strip()
            try:
                mode = int(mode_in)
            except:
                mode = 0

            tokens = tokenize_string(query)
            lists: List[Tuple[int, InvertedList]] = []

            for token in tokens:
                if token not in lexicon:
                    print(f"[WARN] term {token} missing from lexicon", file=sys.stderr)
                    continue
                lex_entry = lexicon[token]
                curr_list = open_inverted_list(lex_entry, index_file)
                lists.append((curr_list.numDocs, curr_list))

            lists.sort(key=lambda x: x[0])


    print("Exiting.")


if __name__ == "__main__":
    main()
