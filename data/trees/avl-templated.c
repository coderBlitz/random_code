#ifndef AVL_C_
#define AVL_C_

#include"avl.h"

static size_t updateHeight(avl_node *root){
	size_t left = 0;
	size_t right = 0;
	if(root == NULL) return 0;

	if(root->left != NULL) left = root->left->height;
	if(root->right != NULL) right = root->right->height;

	root->height = ((left > right)?left:right) + 1;

	return root->height;
}

static void rotateLeft(avl_node **parent, avl_node *child){
	size_t child_size = 0;	// Make C90 happy
	avl_node *oldChild = NULL;

	if(child == NULL) printk(KERN_INFO "Something wrong\n");
	// Child (right) replaces parent node
	// Parent becomes left subtree of child
	// Left subtree of child becomes right subtree of parent

	child_size = child->size;
	child->size = (*parent)->size;
	(*parent)->size -= child_size; // Parent losing all nodes of subtree

	oldChild = child->left;
	if(oldChild) (*parent)->size += oldChild->size;

	child->left = *parent;
	(*parent)->right = oldChild;
	updateHeight(*parent);

	*parent = child;
	updateHeight(child);
}

static void rotateRight(avl_node **parent, avl_node *child){
	size_t child_size = 0;	// C90
	avl_node *oldChild = NULL;

	if(child == NULL) printk(KERN_INFO "Something wrong\n");
	// Child (left) replaces parent node
	// Parent becomes right subtree of child
	// Right subtree of child becomes left subtree of parent

	child_size = child->size;
	child->size = (*parent)->size;
	(*parent)->size -= child_size; // Parent losing all nodes of subtree

	oldChild = child->right;
	if(oldChild) (*parent)->size += oldChild->size;

	child->right = *parent;
	(*parent)->left = oldChild;
	updateHeight(*parent);

	*parent = child;
	updateHeight(child);
}

// Rotation helper that will perform the correct rotation for the given root
static void rotate(avl_node **tree){
	size_t left;
	size_t right;
	int32_t diff;

	size_t rightChild;
	size_t leftChild;
	int32_t childDiff;

	if(tree == NULL || (*tree) == NULL) return;

	left = ((*tree)->left == NULL)?0:(*tree)->left->height;
	right = ((*tree)->right == NULL)?0:(*tree)->right->height;
	diff = left-right;

	if(diff == 2){
		// Check for double rotation condition
		// The end root will be the subtree of the child with larger height
		rightChild = ((*tree)->left->right == NULL)?0:(*tree)->left->right->height;
		leftChild = ((*tree)->left->left == NULL)?0:(*tree)->left->left->height;
		childDiff = rightChild - leftChild;

		if(childDiff > 0){
			// Rotate left-right
			// Rotate left child and left->right child
			rotateLeft(&((*tree)->left), (*tree)->left->right);
			rotateRight(tree, (*tree)->left);
		}else{
			// Normal rotation right
			rotateRight(tree, (*tree)->left);
		}
	}else if(diff == -2){
		rightChild = ((*tree)->right->right == NULL)?0:(*tree)->right->right->height;
		leftChild = ((*tree)->right->left == NULL)?0:(*tree)->right->left->height;
		childDiff = leftChild - rightChild;

		if(childDiff > 0){
			// Rotate right-left
			// Rotate right child and right->left child
			rotateRight(&((*tree)->right), (*tree)->right->left);
			rotateLeft(tree, (*tree)->right);
		}else{
			// Normal rotation left
			rotateLeft(tree, (*tree)->right);
		}
	}else if(diff > 1){
		// Normal rotation
		rotateRight(tree, (*tree)->left);
	}else if(diff < -1){
		rotateLeft(tree, (*tree)->right);
	}
}


// Inserts data into tree, performs rotations and updates heights
int avlInsert(avl_node **tree, avl_id_t id, data_t data){
	if(*tree == NULL){
		(*tree) = kmalloc(sizeof(avl_node), GFP_KERNEL);// Once at end of branch, insert leaf
		if(*tree == NULL){
			return -ENOMEM;
		}
		(*tree)->id = id;
		(*tree)->data = data;
		(*tree)->size = 1;
		(*tree)->height = 1;
		(*tree)->left = NULL;
		(*tree)->right = NULL;

		return 0;
	}else{
		char ret = 0;
		if((*tree)->id == id){
			return -EEXIST; // If data already exists
		}
		else if((*tree)->id > id){
			ret = avlInsert(&((*tree)->left), id, data); // Recurse to add
			if(!ret){
				(*tree)->size++; // Increment only if data did not exist
				updateHeight(*tree);

				// Re-balance
				rotate(tree);
			}
			return ret;
		}else if((*tree)->id < id){
			ret = avlInsert(&((*tree)->right), id, data);
			if(!ret){
				(*tree)->size++; // Increment only if data did not exist
				updateHeight(*tree);

				// Re-balance
				rotate(tree);
			}

			return ret;
		}
	}

	return -1; // Useless return
}

// Return minimum value in tree, and delete it. Val will be additional data
static avl_id_t avlDeleteMin(avl_node **tree, data_t *val){
	avl_id_t ret;

	if((*tree) != NULL){
		if((*tree)->left != NULL){
			ret = avlDeleteMin(&((*tree)->left), val);

			(*tree)->size--;
			updateHeight(*tree);

			// Re-balance
			rotate(tree);

			return ret;
		}else{ // Remove min, only has right child
			avl_node *old = (*tree);
			(*tree) = (*tree)->right; // Move subtree up
			if(*tree != NULL){
				//(*tree)->size--;
				updateHeight((*tree));
				rotate(tree);
			}

			ret = old->id;
			*val = old->data;
			kfree(old);
			old = NULL;

			return ret;
		}
	}

	return 0;
}
static avl_id_t avlDeleteMax(avl_node **tree, data_t *val){
	avl_id_t ret;

	if((*tree) != NULL){
		if((*tree)->right != NULL){
			ret = avlDeleteMax(&((*tree)->right), val);
			// Only update on successful remove
			(*tree)->size--;
			updateHeight(*tree);

			// Re-balance
			rotate(tree);

			return ret;
		}else{ // Remove max, only has left child
			avl_node *old = (*tree);
			(*tree) = (*tree)->left; // Move subtree up
			if(*tree != NULL){
				//(*tree)->size--;
				updateHeight((*tree));
				rotate(tree);
			}

			ret = old->id;
			*val = old->data;
			kfree(old);
			old = NULL;

			return ret;
		}
	}

	return 0;
}

int avlRemove(avl_node **tree, avl_id_t id){
	if((*tree) == NULL) return -ENOENT;

	if((*tree)->id == id){
		// If at the node to remove, get min of right or max of left for root
		// Maybe have it choose larger side to take from
		if((*tree)->right != NULL){
			// Get min value from right subtree to replace root
			data_t val;
			(*tree)->id = avlDeleteMin(&((*tree)->right), &val);
			(*tree)->data = val;
		}else if((*tree)->left != NULL){
			// Get max value from left subtree to replace root
			data_t val;
			(*tree)->id = avlDeleteMax(&((*tree)->left), &val);
			(*tree)->data = val;
		}else{
			// Node is leaf, simply delete
			kfree(*tree);
			(*tree) = NULL;
			return 0;
		}

		(*tree)->size--;
		updateHeight(*tree);
		rotate(tree);
		return 0;
	}
	else if((*tree)->id > id){ // Traversal conditions
		int ret = avlRemove(&((*tree)->left), id);

		// Only update on successful remove
		if(!ret){
			(*tree)->size--;
			updateHeight(*tree);
			rotate(tree);
		}

		return ret;
	}else if((*tree)->id < id){
		int ret = avlRemove(&((*tree)->right), id);
		if(!ret){
			(*tree)->size--;
			updateHeight(*tree);
			rotate(tree);
		}

		return ret;
	}

	return -ENOENT;
}

// Returns non-zero if data is in tree, zero otherwise
int avlFind(avl_node *tree, avl_id_t id, data_t **value){
	if(tree != NULL){
		if(tree->id == id){
			*value = &tree->data;
			return 1; // Found data
		}
		else if(tree->id > id) return avlFind(tree->left, id, value); // Recurse to add
		else if(tree->id < id) return avlFind(tree->right, id, value);
	}

	return 0;// Not found (false)
}

size_t avlSize(const avl_node *tree){
	if(tree != NULL){
		int count = avlSize(tree->left);
		count++; // This node
		count += avlSize(tree->right);

		return count;
	}

	return 0;
}

size_t avlMaxHeight(const avl_node *root){
	int left = 0;
	int right = 0;
	if(root == NULL){
		return 0;
	}

	if(root->left != NULL){
		left = avlMaxHeight(root->left) + 1;
	}

	if(root->right != NULL){
		right = avlMaxHeight(root->right) + 1;
	}

	if(left > right) return left;
	else return right;

	return 0;
}

avl_id_t avlMax(const avl_node *tree){
	if(tree == NULL) return 0;
	if(tree->right == NULL) return tree->id;// If next node is null, max
	return avlMax(tree->right);
}
avl_id_t avlMin(const avl_node *tree){
	if(tree == NULL) return 0;
	if(tree->left == NULL) return tree->id;// If next node is null, min
	return avlMin(tree->left);
}

void destroy(avl_node **tree){
	if((*tree) != NULL){
		if((*tree)->left != NULL) destroy(&((*tree)->left));
		if((*tree)->right != NULL) destroy(&((*tree)->right));
		kfree(*tree);
		*tree = NULL;
	}
}

#endif
