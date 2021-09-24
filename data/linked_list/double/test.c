#include<stdio.h>
#include<stdlib.h>
#include"node.h"

int main(){
	node *head = NULL;// Tracks last element in
	node *tail;// Tracks first element in

	tail = insert(&head,5);// If we remember the first element in, it becomes tail
	tail->data = 4;// Shows we can change data
	insert(&head,6);
	insert(&head,7);
	insert(&head,8);

	traverse(head);// Front-to-back

	delete(&head,0);
	delete(&tail,0);

	printf("After 2 gone\n");
	traverse(head);

	delete(&tail,0);
	delete(&head,0);
	printf("\n");

	printf("Head\n");
	traverse(head);
	printf("Tail\n");
	traverse(tail);// Back-to-front
}
