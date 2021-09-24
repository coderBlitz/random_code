#ifndef BTREE_H
#define BTREE_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

typedef long btree_data_t;

// This is public struct, which is used to hold the degree (primarily) and total size
struct btree{
	unsigned short degree; // How big arrays are
	size_t size; // Size of entire tree
	struct btreeNode *root;
};


void btree_destroy(struct btree *);

/**	Public function, with private inside c file
**/
int btree_insert(struct btree *, const btree_data_t);

/**	Delete/remove 'data' from the tree.
**/
int btree_delete(struct btree *bt, const btree_data_t data);

/**	Returns nonzero if data exists in tree.
**/
int btree_find(struct btree const *bt, const btree_data_t);

void btree_print(struct btree const *);

#endif
