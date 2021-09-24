#include"btree.h"

struct btreeNode{
	btree_data_t *data; // Data array
	struct btreeNode **nodes; // Child node array
	size_t size; // Amount of data in this node
};

void _btree_destroy(struct btreeNode **bt, const unsigned short degree){
	if(bt == NULL || *bt == NULL) return;

	// Recursively delete all child nodes
	for(int i = 0;i <= (*bt)->size;i++){
		_btree_destroy((*bt)->nodes + i, degree);
		(*bt)->nodes[i] = NULL;
	}

	// Delete data and node arrays
	free((*bt)->data);
	free((*bt)->nodes);
	free(*bt);
	(*bt) = NULL;
}

void btree_destroy(struct btree *bt){
	if(bt == NULL) return;

	_btree_destroy(&bt->root, bt->degree);
}

/**	Allocation helper for btreeNode, since error code is large
**/
struct btreeNode *create_btree_node(unsigned short degree){
	struct btreeNode *ret = NULL;
	void *tmp = NULL;
	// Alloc node
	tmp = malloc(sizeof(*ret));
	if(tmp == NULL){
		return NULL;
	}
	ret = tmp;

	// Alloc data array (degree+1) for extra slot
	tmp = calloc((degree+1), sizeof(*ret->data));
	if(tmp == NULL){
		return NULL;
	}
	ret->data = tmp;

	// Alloc nodes array as NULLs (degree+1+1) for extra slot (in addition to normal +1)
	tmp = calloc((degree+2), sizeof(*ret->nodes));
	if(tmp == NULL){
		return NULL;
	}
	ret->nodes = tmp;

	// Set values
	ret->size = 0;

	return ret;
}

/**
Let degree be amount of data per node (not including extra 1 at end of each array)
TODO: Try to use binary search for data position
Process:
	***** Finding data index *****
	While new data greater than data at indices [0,size)
		increment
	***** Determine if recursing or inserting *****
	If child node at stop index is NULL
		Insert data at stop index
		Increment (this node) size
	Else
		Recurse to child node at stop index
		If return greater than 0
			***** Handling pushed up data *****
			Shift data at indices [stop index, size) forward by 1
			Shift node pointers at indices [stop index + 1, size] forward by 1
			Place lifted data at stop index (new gap index)
			Create new node (pointer) at stop index + 1
			Copy back half (data & nodes) of (child node at stop index) to (new node pointer at stop index + 1) and set new size
			Set new size of (child node at stop index), as no data is moved
	If new leaf size > degree
		Push data at floor(new size/2)
		Return stop index+1
	Else
		Return 0

	lift -- data that needs to be pushed up to parent
	Returns negative for error, strictly positive (> 0) for lift data
**/
int _btree_insert(struct btreeNode *bt, unsigned short degree, const btree_data_t data, btree_data_t *lift){
	if(bt == NULL) return 0;

	int res = 0;
	int stop = 0;
	while(stop < bt->size && data > bt->data[stop]){
		stop++;
	}

	//printf("Stop index: %2d\n", stop);
	if(data == bt->data[stop]){
		//printf("Can't insert duplicate data = %ld.\n", data);
		return -1;
	}

	if(bt->nodes[stop] == NULL){
		// Insert here at "leaf"
		//printf("Inserting %ld at %p\n", data, bt);

		// Shift data and nodes right
		if(stop < degree){
			//printf("Shifting data from %d to %d for %4lu bytes\n", stop, stop+1, (bt->size-stop) * sizeof(*bt->data));
			memmove(bt->data + stop + 1, bt->data + stop, (bt->size-stop) * sizeof(*bt->data));
		}
		bt->data[stop] = data;

		bt->size++; // Increment size
	}else{
		// Recurse
		//printf("Recursing\n");
		res = _btree_insert(bt->nodes[stop], degree, data, lift);
		if(res > 0){
			/* Cleave node
			*/
			//printf("Cleaving node at %2d with data %ld\n", stop, *lift);
			struct btreeNode *tmp;

			// Shift data and nodes right
			if(stop < degree){
				//printf("Shifting data from %d to %d for %4lu bytes\n", stop, stop+1, (bt->size-stop) * sizeof(*bt->data));
				memmove(bt->data + stop + 1, bt->data + stop, (bt->size-stop) * sizeof(*bt->data));
			}
			if((stop+1) <= degree){
				//printf("Shifting nodes from %d to %d for %4lu bytes\n", stop+1, stop+2, (1+bt->size-stop) * sizeof(*bt->nodes));
				memmove(bt->nodes + stop + 2, bt->nodes + stop + 1, (bt->size-stop) * sizeof(*bt->nodes));
			}

			// Place new data
			bt->data[stop] = *lift;

			// Create new right node
			tmp = create_btree_node(degree); // tmp is new right node
			if(tmp == NULL){
				return -1;
			}
			bt->nodes[stop+1] = tmp;
			//printf("New right node = %p @ %2d\n", tmp, stop+1);

			// Copy back half data to new right node from original node, and set size
			struct btreeNode *old = bt->nodes[stop];
			//printf("Old = %p @ %2d\n", old, stop);
			size_t back = old->size - res - 1;
			//printf("Moving data and nodes from %2d for %3d and %3d units\n", res+1, back, back+1);
			memcpy(tmp->data, old->data + (res + 1), (back) * sizeof(*old->data)); // NEW
			memcpy(tmp->nodes, old->nodes + (res + 1), (back+1) * sizeof(*old->nodes)); // NEW

			// Update new and old node sizes
			//printf("Old size: %2ld\n", old->size);
			tmp->size = (old->size - 1)/2;
			old->size /= 2; // Old size simply halved (rounding in favor)

			bt->size++; // Increment this node size and reset return
			res = 0;
		}
	}

	// Check if node full ==> push median value up
	if(bt->size > degree){
		*lift = bt->data[bt->size / 2];
		//printf("Pushing up data = %ld\n", *lift);

		res = bt->size / 2;
	}

	return res;
}

/**	Insert node and handle pushed data, which could create new root.
	Will not accept degree 1 B-Tree, because it is pointless (and breaks find)
**/
int btree_insert(struct btree *bt, const btree_data_t data){
	if(bt == NULL || bt->degree < 2) return -1;

	struct btreeNode *tmp;
	// If root nodes doesn't exist, create it
	if(bt->root == NULL){
		//printf("Creating root..\n");
		// Alloc node
		tmp = create_btree_node(bt->degree);
		if(tmp == NULL){
			return -1;
		}
		bt->root = tmp;

		// Set values
		bt->root->size = 1;
		bt->root->data[0] = data;
		bt->size = 1;

		return 0;
	}

	// Call recursive insert, and give parameter to push data with
	btree_data_t up;
	int res = _btree_insert(bt->root, bt->degree, data, &up);

	// If error, return immediately. If success, increment size and return.
	if(res < 0) return res;
	if(res == 0){
		bt->size++;
		return res;
	}

	/* Handle pushed data by creating new root, splitting old root in twain
	*/
	//printf("Res = %3d\tpushed data = %ld\n", res, up);
	//btree_print(bt);

	// Create new root
	tmp = create_btree_node(bt->degree);
	if(tmp == NULL){
		return -1;
	}

	// Insert data to new root node
	tmp->data[0] = up;
	tmp->size = 1;

	// Assign old root
	tmp->nodes[0] = bt->root;
	bt->root = tmp;

	// Create new right node
	tmp = create_btree_node(bt->degree); // tmp is new right node
	if(tmp == NULL){
		return -1;
	}
	bt->root->nodes[1] = tmp;

	// Copy back half data/nodes to new right node from original node, and set size
	struct btreeNode *old = bt->root->nodes[0];
	size_t back = old->size - res - 1;
	//printf("Copying back half data starting at %d for %4lu units\n", res, back);
	memcpy(tmp->data, old->data + res+1, (back) * sizeof(*old->data));
	memcpy(tmp->nodes, old->nodes + res+1, (back+1) * sizeof(*old->nodes));

	// Update new and old node sizes
	tmp->size = (old->size - 1)/2;
	old->size /= 2; // Old size simply halved (rounding in favor)

	bt->size++; // Increment tree size

	return 0;
}

/**	Linearly search each node to find data, or child to recurse to
**/
int _btree_find(struct btreeNode const* bt, const unsigned short degree, const btree_data_t data){
	if(bt == NULL) return 0;

	// Binary search
	long a = 0, b = bt->size, c = (b-a)/2;
	do{
		if(data < bt->data[c]){
			b = c;
		}else if(bt->data[c] < data){
			a = c + 1;
		}
		c = a + (b-a)/2;
	}while(a < b && bt->data[c] != data);

	if(bt->data[c] == data){
		return 1;
	}else if(bt->nodes[c] == NULL){
		return 0;
	}

	// Recurse
	return _btree_find(bt->nodes[c], degree, data);
}

int btree_find(struct btree const *bt, const btree_data_t data){
	return _btree_find(bt->root, bt->degree, data);
}

int _btree_delete(struct btreeNode *bt, const unsigned short degree, const btree_data_t data){
	if(bt == NULL) return 0;

	int res = -1;
	int stop = 0;
	while(stop < bt->size && bt->data[stop] < data){
		stop++;
	}

	if(bt->data[stop] == data){
		// Delete from this node
		printf("Found data!\n");
		res = 0;
	}else if(bt->nodes[stop] != NULL){
		// Merge/steal (maybe) and recurse
		printf("Recursing to node %d\n", stop);
		res = _btree_delete(bt->nodes[stop], degree, data);
	}

	return res;
}

int btree_delete(struct btree *bt, const btree_data_t data){
	return _btree_delete(bt->root, bt->degree, data);
}

void _btree_print(struct btreeNode *const bt, const unsigned short degree){
	if(bt == NULL) return;

	printf("%p data:\n[", bt);
	for(int i = 0;i < bt->size;i++){
		printf("%ld ", bt->data[i]);
	}
	printf("]\n");

	// Recurse print
	for(int i = 0;i <= bt->size;i++){
		printf("%p Node %3d:\n", bt, i);
		_btree_print(bt->nodes[i], degree);
		//printf("------\n");
	}
}

void btree_print(struct btree const *bt){
	if(bt == NULL) return;

	_btree_print(bt->root, bt->degree);
}
