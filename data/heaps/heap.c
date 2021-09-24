#include"heap.h"

/**	Shift data, at index 'child', up the heap.
**/
size_t heap_trickle_up(struct heap *hp, size_t child, void *data){
	if(hp == NULL) return 0;

	/* Move new data up the tree
	*/
	size_t idx = child / 2;

	// While the parent is in the tree (not 0) and values should shift, swap parent and child
	while(idx && hp->pri(data, hp->data[idx])){
		hp->data[child] = hp->data[idx]; // Shift parent down

		// Move up in the heap
		child = idx;
		idx /= 2;
	}

	// Data goes at the emptied position (from swapping values down)
	hp->data[child] = data;

	return child;
}

/**	Shift data, at index 'idx', down the heap
**/
size_t heap_trickle_down(struct heap *hp, size_t idx, void *data){
	if(hp == NULL) return 0;

	/* Move data down the tree
	*/
	size_t child = 2 * idx;
	while(child <= hp->size){
		// If there are left and right children, pick which one
		if(child < hp->size) child = hp->pri(hp->data[child], hp->data[child+1]) ? child : (child + 1);

		//printf("idx: %X %lu\tchild: %X %lu\n", hp->data[idx], idx, hp->data[child], child);
		if(hp->pri(data, hp->data[child])) break;

		// Shift values
		hp->data[idx] = hp->data[child];

		idx = child;
		child *= 2;
	}

	// Place data in correct spot
	hp->data[idx] = data;

	return idx;
}

/**	Zero members, assign priority, and allocate rounded size
**/
int heap_init(struct heap *hp, size_t capacity, int (*priority)(void *, void *)){
	if(hp == NULL || priority == NULL || capacity == 0) return -1;

	hp->size = 0;
	hp->capacity = 1 << (32 - __builtin_clzl(capacity));
	//printf("Assigned capacity: %lu\n", hp->capacity);
	hp->pri = priority;

	hp->data = malloc(hp->capacity * sizeof(*hp->data));
	if(hp->data == NULL){
		return -1;
	}
}

/**	Zero everything and free the array
**/
void heap_destroy(struct heap *hp){
	if(hp == NULL) return;

	hp->size = 0;
	hp->capacity = 0;
	free(hp->data);
	hp->data = NULL;
}

/**	Allocate space, then shift data in heap until new data is ordered
**/
int heap_insert(struct heap *hp, void *data){
	if(hp == NULL) return -1;

	++hp->size;

	// Increase capacity as necessary
	if(hp->size > hp->capacity){
		hp->capacity *= 2;
		void **temp = realloc(hp->data, hp->capacity * sizeof(*hp->data));
		if(temp == NULL){
			perror("Realloc failed");
			return -1;
		}
		hp->data = temp;
	}

	
	/* Move new data up the tree
	*/

	heap_trickle_up(hp, hp->size, data);

	return 0;
}

void * heap_pop(struct heap *hp){
	// Grab end item, replace root, trickle down
	if(hp == NULL || hp->size == 0) return NULL;

	// Get return, fetch last entry, and decrease size
	void *ret = hp->data[1];
	void *data = hp->data[hp->size];
	--hp->size;

	// If 0, do not shift
	if(hp->size < 1) return ret;

	/* Move data down the tree
	*/

	heap_trickle_down(hp, 1, data);

	return ret;
}

/**	Moves the value up or down as needed (assumes the priority has changed)
**/
int heap_update(struct heap *hp, void *data){
	if(hp == NULL || hp->size == 0) return -1;

	// Find data location
	size_t idx = 0;
	for(size_t i = 1;i <= hp->size;++i){
		if(hp->data[i] == data){
			idx = i;
			break;
		}
	}

	if(idx == 0) return -1;

	//printf("Found data at index: %lu\n", idx);

	/* Move new data up the tree, return early if moved.
	*/
	if(heap_trickle_up(hp, idx, data) != idx){
		return 0;
	}

	/* Move new data down the tree.
	*/
	heap_trickle_down(hp, idx, data);

	return 0;
}

int heap_clear(struct heap *hp){
	if(hp == NULL) return -1;

	hp->size = 0;
}

void heap_dump(struct heap *hp){
	if(hp == NULL) return;

	printf("Size: %lu\tCapacity: %lu\n", hp->size, hp->capacity);
	for(long i = 1;i <= hp->size;++i){
		printf("[%3d]:\t0x%08lX\n", i, hp->data[i]);
	}
}
