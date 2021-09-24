#ifndef HEAP_H
#define HEAP_H

#include<errno.h>
#include<stdio.h>
#include<stdlib.h>

struct heap{
	size_t size;
	void **data; // Could use index 0 as size if desired
	size_t capacity;
	int (*pri)(void *, void *);	// Priority function
};

/**	Returns true if a < b
**/
static inline int HEAP_MIN(void *a, void *b){
	return (a < b);
}

/**	Returns true if a > b
**/
static inline int HEAP_MAX(void *a, void *b){
	return (a > b);
}

/**	Allocate initial capacity as next power of 2, and assign priority function.
**/
int heap_init(struct heap *, size_t, int (*)(void *, void *));
void heap_destroy(struct heap *);
int heap_insert(struct heap *, void *);
void * heap_pop(struct heap *);
int heap_update(struct heap *, void *);
int heap_clear(struct heap *);
void heap_dump(struct heap *);

#endif
