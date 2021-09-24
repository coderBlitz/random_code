#ifndef SQUARELIST_H
#define SQUARELIST_H

struct SquareList{
	uint64_t size;
	uint32_t width;
	int64_t **data;
};

void insert(int64_t item);

#endif
