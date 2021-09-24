/*	sorted_vec.c -- A dynamically growing array, with sorted data.
Notes:
	Like the c++ vector class. Doubles capacity each reallocation.
*/
#include"sorted_vec.h"
#include<stdio.h>
#include<stdlib.h>

/** Add atExit() function and array to keep track of all allocated structs
	  and then use the fn to de-allocate for cleanup
**/

struct sorted_buffer{
	uint64_t size;
	uint64_t capacity;
	uint64_t start;
	uint64_t end;
	int64_t *data;
};

SortedBuffer newBuffer(const uint64_t capacity){
	if(capacity == 0) return NULL;

	struct sorted_buffer *new_buf = malloc(sizeof(*new_buf));
	new_buf->size = 0;
	new_buf->capacity = capacity;
	new_buf->start = 0;
	new_buf->end = 0;

	new_buf->data = malloc(capacity * sizeof(*new_buf->data));
	for(uint64_t i = 0;i < new_buf->size;++i) new_buf->data[i] = 0;

	return new_buf;
}

int8_t freeBuffer(const SortedBuffer buf){
	if(buf != NULL){
		struct sorted_buffer *buffer = (struct sorted_buffer *)buf;
		if(buffer->data != NULL) free(buffer->data);
		free(buffer);

		return 0;
	}

	return -1;
}

int8_t insertBuffer(const SortedBuffer buf,const int64_t data){
	// Insert sorted
	if(buf == NULL) return -1;
	struct sorted_buffer *buffer = (struct sorted_buffer *) buf;
	if(buffer->data == NULL || buffer->size == buffer->capacity) return -1;

	uint64_t capacity = buffer->capacity;
	uint64_t start = buffer->start;
	uint64_t last = (capacity + buffer->end - 1)%capacity;

	if(buffer->size == 0){
		buffer->data[start] = data;
		buffer->end = (last+2)%capacity;
	}else if(data > buffer->data[last]){
		// New max
		buffer->data[buffer->end] = data;
		buffer->end = (last+2)%capacity;
	}else if(data < buffer->data[start]){
		// New min
		start = (capacity + start - 1)%capacity;
		buffer->data[start] = data;
		buffer->start = start;
	}else{
		// Somewhere in the middle
		uint8_t shifting = 0;
		int64_t temp = 0;
		uint64_t idx = 0;
		for(uint64_t i = 0;i <= buffer->size;++i){
			idx = (i+start)%capacity;
			if(shifting){
				temp ^= buffer->data[idx];
				buffer->data[idx] ^= temp;
				temp ^= buffer->data[idx];
			}else{
				if(buffer->data[idx] > data){
					shifting = 1;
					temp = buffer->data[idx];
					buffer->data[idx] = data;
				}
			}
		}

		buffer->end = (last+2)%capacity;
	}

	buffer->size += 1;
	return 0;
}

int64_t searchBuffer(const SortedBuffer buf,const int64_t data){
	if(buf == NULL) return -1;
	struct sorted_buffer *buffer = (struct sorted_buffer *) buf;
	if(buffer->size == 0 || buffer->data == NULL) return -1;

	uint64_t capacity = buffer->capacity;

	// Check if data less than start (min), not found if true
	if(buffer->start == buffer->end){
		if(data > buffer->data[(buffer->capacity + buffer->end - 1)%buffer->capacity]) return -1;
	}else if(data > buffer->data[buffer->end]) return -1;
	if(data < buffer->data[buffer->start]) return -1;

	for(uint64_t i = 0;i < capacity;++i) if(buffer->data[i] == data) return i; // Linear search

	uint64_t width = capacity;
	uint64_t idx = width >> 1;
	uint64_t upper = capacity-1;
	uint64_t lower = 0;
	uint64_t count = 0;
	uint64_t start = buffer->start;
/*
	register int64_t current = 0;
	while(width > 0 && upper >= lower){

		width = upper-lower;
		current = buffer->data[idx];
		printf("\tIdx: %d (%d)\tWidth: %d\tup: %d\tlow: %d\n", idx, current, width, upper, lower);

		if(current == data) return idx;

		if(current > data){
			upper = idx;
		}
		if(current < data){
			lower = (idx+1;
		}
		idx = ((upper+lower)/2 + start)%capacity;
	}
*/
	return -1;
}

void printBuffer(const SortedBuffer buf){
	if(buf == NULL) return;
	struct sorted_buffer *buffer = (struct sorted_buffer *) buf;

	printf("Size: %lu\tCapacity: %lu\nStart: %lu\tEnd: %lu\n", buffer->size, buffer->capacity, buffer->start, buffer->end);
	for(uint64_t i = 0;i < buffer->capacity;++i) printf("[%lu]: %ld\n", i, buffer->data[i]);
}

void fill(SortedBuffer buf){
	struct sorted_buffer *buffer = (struct sorted_buffer *) buf;
	for(uint64_t i = 0;i < buffer->capacity;++i) buffer->data[i] = i+1;
	buffer->size = buffer->capacity;
}
