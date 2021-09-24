#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"node.h"

#define DEBUG 0


node *XOR(struct node *a, struct node *b){
	return (struct node *)((uintptr_t)(a) ^ (uintptr_t)(b));
}

node* insert(node** start, int data){// Create a node
	node *new_node = (node*)malloc(sizeof(node));// Memory allocate node
	if(new_node == NULL){
		printf("Could not create node with data %d\n",data);
		return NULL;
	}

	new_node->data = data;
	new_node->both = XOR(*start, NULL);

	if(*start != NULL){
		node *tail = XOR((*start)->both, NULL);
		(*start)->both = XOR(tail, new_node);
	}

	*start = new_node;

	return new_node;
}

void delete(node **start, int N){
	node *current = *start;
	node *prev = NULL;
	node *next;
	int count = 0;
	while(current != NULL && count != N){
//		printf("Node #%d: %d\n",count++,current->data);
		next = XOR(prev, current->both);

		prev = current;
		current = next;

		count++;
	}
//	printf("Will delete node #%d, with data %d\n",count,current->data);

	// XOR current and next for head pointer, XOR current and prev for tail pointer
	// XOR head and tail for new node->both value
	// Order of variable nodes, head-prev-current-next-tail

	next = XOR(current->both, prev);// Next is O-O-C-X-O

	if(DEBUG) printf("\nPrev: %X\nCurrent: %X\nNext: %X\n\n", prev, current, next);
	if(N == 0) (*start) = next;// If we delete the first element, we need to update the head

	node *tail = NULL;
	if(next != NULL) tail = XOR(current, next->both);// Tail is O-O-C-O-X

	node *head = NULL;
	if(prev != NULL) head = XOR(current, prev->both);// Head is X-O-C-O-O

	if(DEBUG) printf("Head: %X\nTail: %X\n\n",head,tail);

	if(prev != NULL) prev->both = XOR(head, next);
	if(DEBUG) printf("Done prev\n");

	if(next != NULL) next->both = XOR(tail, prev);
	if(DEBUG) printf("Done next\n");

	// Delete after here
	free(current);
}

void traverse(node* start){// Go through and print list
	node *current = start;
	node *prev = NULL;
	node *next;
	int count = 0;
	while(current != NULL){
		if(DEBUG) printf("Node #%d: %d -- %X\n",count++,current->data,current);
		else printf("Node #%d: %d\n",count++,current->data);

		next = XOR(prev, current->both);

		prev = current;
		current = next;
	}
}
