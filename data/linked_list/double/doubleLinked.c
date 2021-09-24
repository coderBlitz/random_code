#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#define DEBUG 0

typedef struct node{
	int data;// Holds the actual data (who would have though)
	struct node *both;// Points to next node
} node;

node* XOR(struct node *a, struct node *b){// Helper function
	return (node *)((uintptr_t)(a) ^ (uintptr_t)(b));
}

node* insert(int data, node **head){// Insert can prepend or append, depending on the pointer given
	node *new_node = (node *) malloc(sizeof(node));// Create new node pointer

	if(new_node == NULL){// Failed to malloc
		fprintf(stderr,"Error creating new node.\n");
		exit(1);
	}
	if(DEBUG)	printf("Creating node with data %d\n",data);

	new_node->data = data;// Set data importantly
	new_node->both = XOR(*head, NULL);

	if(*head != NULL){
		node *tail = XOR((*head)->both, NULL);
		(*head)->both = XOR(new_node, tail);
	}
	if(DEBUG) printf("Created node with data %d\n",new_node->data);
	*head = new_node;// Sets new head node

	return new_node;
}

void delete(int N, node **start){
	node *current = *start;
	node *prev = NULL;
	node *next;
	long count = 0;
	while(current != NULL && count != N){
//		printf("Node #%ld: %d\n", count++, current->data);
		next = XOR(prev, current->both);

		prev = current;
		current = next;

		count++;
	}
//	printf("Will delete node #%d, with data %d\n",count, current->data);

	// XOR current and next for tail, and XOR current and prev for head pointer
	// XOR those
	next = XOR(current->both, prev);// Actually the next node

	if(DEBUG) printf("\nPrev; %X\nCurrent: %X\nNext: %X\n", prev, current, next);
	if(N == 0) (*start) = next;// If we delete the first element, we need to update the head

	node *tail = NULL;
	if(next != NULL) tail = XOR(current, next->both);

	node *head = NULL;
	if(prev != NULL) head = XOR(current, prev->both);

	if(DEBUG) printf("Head: %X\nTail: %X\n", head, tail);

	if(prev != NULL) prev->both = XOR(head, next);
	if(DEBUG) printf("Prev done\n");

	if(next != NULL) next->both = XOR(tail, prev);
	if(DEBUG) printf("Head done\n");

	// Delete node
	free(current);
}

void traverse(node *head){// Goes through list, direction depends on pointer given
	node *current = head;
	node *prev = NULL;
	node *next;
	long count = 0;
	while(current != NULL){
		printf("Node #%ld: %d\n", count++, current->data);

		next = XOR(prev, current->both);

		prev = current;
		current = next;
	}
}

int main(){
	node *head = NULL;
	node *tail = NULL;

	tail = insert(4, &head);
	insert(5, &head);
	insert(6, &head);
	insert(7, &head);
	insert(8, &tail);

	traverse(head);
	printf("\n");

	delete(2, &head);

	traverse(head);
}
