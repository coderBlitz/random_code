#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"bst.h"

Node * singleRight(Node *root){
	for(int i = 0;i < 500;++i){
		insert(&root, i);
	}

	return root;
}

int checkBST(Node *root){
	if(root == NULL){
		printf("Empty root\n");
		return 0;
	}

	int good = 0;

	if(root->left == NULL && root->right == NULL){
		if(root->size != 1){
			printf("Leaf 0x%X has invalid size of %d\n", root->size);
			good = -1;
		}
	}

	if(root->left != NULL && root->right == NULL){ // Check left subtree
		if(root->left->data > root->data){
			printf("Left 0x%X has larger data than 0x%X. %d !< %d\n", root->left, root, root->left->data, root->data);
			good = -1;
		}else if(root->left->data == root->data){
			printf("Left 0x%X has duplicate data of 0x%X. %d\n", root->left, root, root->data);
			good = -1;
		}

		good = checkBST(root->left) || good;
	}

	if(root->right != NULL && root->left == NULL){ // Check right subtree
		if(root->right->data < root->data){
			printf("Right 0x%X has smaller data than 0x%X. %d !< %d\n", root->right, root, root->right->data, root->data);
			good = -1;
		}else if(root->right->data == root->data){
			printf("Right 0x%X has duplicate data of 0x%X. %d\n", root->right, root, root->data);
			good = -1;
		}

		good = checkBST(root->right) || good;
	}

	// Both children non-null, check size
	uint32_t left = (root->left == NULL)?0:root->left->size;
	uint32_t right = (root->right== NULL)?0:root->right->size;
	int combo = left + right;
	if(combo != (root->size - 1)){
		printf("0x%X size %d != %d + %d + 1 = %d\n", root, root->size, root->left->size, root->right->size, combo+1);
		good = -1 || good;
	}

	return good;
}

void repeatedRemoval(Node *tree, int N){
	printf("Testing %d root re-insertions\n", N);

	int data = 0;
	int count = 0;
	for(int i = 0;i < N;++i){
		data = tree->data;
		if(removeNode(&tree, tree->data) == 0){
			insert(&tree, data);
		}else ++count;
	}

	if(count) printf("%d re-insertions failed\n");
}

int main(){
	srand(time(0));

	Node *test = NULL;
	int N = 1000;
	int mod = N*3;
	int nums[N];

	printf("Inserting %d numbers into test..\n",N);
	for(int i = 0;i < N;i++){
		nums[i] = rand() % mod;
		//printf("Inserting %d\n", nums[i]);
		if(insert(&test, nums[i])){
			//printf("%d is a duplicate\n", nums[i]);
			--i;
		}
	}

	//test = singleRight(test);
	printf("\nRoot is (0x%X)\n",test);
	//printTree(test);

	printf("Tree size after insert: %u\tnode_size = %u\n", size(test), (test == NULL)?0:test->size);
	if(test != NULL && test->left != NULL){
		printf("Left-size: %u/%u\n", test->left->size, size(test->left));
	}
	if(test != NULL && test->right != NULL){
		printf("Right-size: %u/%u\n", test->right->size, size(test->right));
	}

	// Check tree structure
	printf("\nChecking tree..\n");
	int ret = checkBST(test);
	printf("Tree is %s\n\n", (ret)?"bad":"good");
	//printf("\nTrying to remove root node test(%d)..\n", test->data);

	printf("Max depth: %u\n", maxHeight(test));

	// Remove shit
	/*printf("Removing everything..\n");
	for(int i = 0;i < N;i++){
		//printf("Attempting to remove %d..\n", nums[i]);
		if(removeNode(&test, nums[i])){
			printf("Tree failed to remove %d\n", nums[i]);
		}
	}*/
	//removeNode(&test, test->data); // Remove root node

	//printf("\nPrinting 0x%x after remove:\n", test);
	//printTree(test);

	repeatedRemoval(test, N/4);

	printf("\nChecking tree..\n");
	ret = checkBST(test);
	printf("Tree is %s\n\n", (ret)?"bad":"good");

	printf("Max depth: %u\n", maxHeight(test));

	printf("Tree size after re-insertions: %u\tnode_size = %u\n", size(test), (test == NULL)?0:test->size);
	if(test != NULL && test->left != NULL){
		printf("Left-size: %u/%u\n", test->left->size, size(test->left));
	}
	if(test != NULL && test->right != NULL){
		printf("Right-size: %u/%u\n", test->right->size, size(test->right));
	}

	printf("Destroying..\n");
	destroy(&test);
}
