#ifndef AVL_H_
#define AVL_H_

#include<linux/errno.h>
#include<linux/kernel.h>
#include<linux/types.h>

#ifndef AVL_DATA_TYPE
	#define AVL_DATA_TYPE long
#endif

typedef AVL_DATA_TYPE data_t;
typedef long avl_id_t;

typedef struct avl_node{
	avl_id_t id;
	data_t data;
	size_t size;			// Size of subtree, including this node (No longer broken)
	size_t height;
	struct avl_node *left;
	struct avl_node *right;
} avl_node;

// Return  0 if inserted, non-zero otherwise
int avlInsert(avl_node **, avl_id_t id, data_t data);

// Return 0 if removed, non-zero otherwise
int avlRemove(avl_node **, avl_id_t id);

// Return non-zero if found, 0 otherwise
int avlFind(avl_node *, avl_id_t id, data_t **val);

// Return max/min number
avl_id_t avlMax(const avl_node *);
avl_id_t avlMin(const avl_node *);

// Get total number of items in tree
size_t avlSize(const avl_node *);

size_t avlMaxHeight(const avl_node *);

// Free up entire tree
void destroy(avl_node **);

#include"avl.c"	// Included for same reason C++ templated classes are.

#endif
