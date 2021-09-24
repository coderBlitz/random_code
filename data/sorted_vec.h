#ifndef SORTED_VEC_H
#define SORTED_VEC_H

#include<stdint.h>

typedef void* SortedBuffer;

SortedBuffer newBuffer(const uint64_t);
int8_t freeBuffer(const SortedBuffer);
int8_t insertBuffer(const SortedBuffer, const int64_t);
int64_t searchBuffer(const SortedBuffer, const int64_t);
void printBuffer(const SortedBuffer);
void fill(SortedBuffer);

#endif
