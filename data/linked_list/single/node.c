#include<stdio.h>
#include<stdlib.h>

#define DEBUG 0

typedef struct node{
	int data;// Holds the actual data (who would have though)
	struct node *next;// Points to next node
} node;

node* createNode(int data, node *next){
	node *new_node = (node *) malloc(sizeof(node));// Create new node pointer
	if(new_node == NULL){// Failed to malloc
		fprintf(stderr,"Error creating new node.\n");
		exit(1);
	}
	if(DEBUG)	printf("Creating node with data %d\n",data);
	new_node->data = data;// Set data importantly
	new_node->next = next;// Points tail to any nodes created before it

	return new_node;
}

node* prepend(int data, node **head){ // Pointer to pointer node so head variable is changed automatically
	node *new_node = createNode(data, *head);
	if(DEBUG) printf("Created node with data %d\n",new_node->data);
	*head = new_node;// Sets new head node
	return *head;// Returns just for the sake of because it may be useful a long time from now in a galaxy far far away
}

node* append(int data, node *head){
	node *tmp = head;
	while(tmp->next != NULL) tmp = tmp->next;
	node *new_node = createNode(data, NULL);
	tmp->next = new_node;
	return new_node;
}

void traverse(node *head){// Goes through list front to back (First-in last-out)
	node *tmp = head;
	long count = 0;
	while(tmp != NULL){
		printf("Node #%ld: %d\n",count++,tmp->data);
		tmp = tmp->next;
	}
}

int main(){
	node *head = NULL;
	prepend(4, &head);
	prepend(5, &head);
	append(6, head);

	traverse(head);
}
