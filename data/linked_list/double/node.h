#include<stdio.h>
#include<stdlib.h>

#ifndef NODE_H_
#define NODE_H_

typedef struct node{// XOR linked list (one pointer for head and tail)
	int data;
	struct node* both;
} node;

node *insert(node**,int);
void delete(node**,int);
void traverse(node*);// Prints list (beginning to end, Last in First out)

#endif
