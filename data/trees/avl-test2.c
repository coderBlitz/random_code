#include<stdio.h>
#include<stdlib.h>

struct test{
	long id;
	int thing;
};

#pragma message "Main"
#define AVL_DATA_TYPE struct test
typedef AVL_DATA_TYPE data_t;
#include"avl.h"


int main(){
	Node *tree = NULL;

	uint32_t N = 10;
	struct test *arr = malloc(N * sizeof(*arr));
	if(arr == NULL) return -1;

	for(uint32_t i = 0;i < N;++i){
		arr[i].id = N-i;
		arr[i].thing = rand();

		avlInsert(&tree, arr[i]);
		//printTree(tree);
		//printf("\n");
	}
	printTree(tree);

	struct test A;
	A.id = 4;
	A.thing = 0;

	struct test *B = &A;
	printf("B = %p\n&B = %p\n", B, &B);
	avlFind(tree, A, &B);


	free(arr);

	return 0;
}
