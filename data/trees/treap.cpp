#ifndef TREAP_CPP
#define TREAP_CPP

#include<cstdlib>
#include<iostream>
#include"treap.h"

template <class T>
Treap<T>::Treap(){
	std::cout << "Empty constructor" << std::endl;
	this->root = NULL;
	this->size = 0;
}

template <class T>
Treap<T>::~Treap(){
	std::cout << "Empty destructor" << std::endl;
	if(root != NULL){
		if(root->left != NULL){
			delete_help(root->left);

			delete root->left;
			root->left = NULL;
		}
		if(root->right != NULL){
			delete_help(root->right);

			delete root->right;
			root->right = NULL;
		}

		delete this->root;
		this->root = NULL;
	}
}
template <class T>
void Treap<T>::delete_help(Node<T> *root){
	if(root->left != NULL){
		delete_help(root->left);

		delete root->left;
		root->left = NULL;
	}
	if(root->right != NULL){
		delete_help(root->right);

		delete root->right;
		root->right = NULL;
	}
}

template <class T>
bool Treap<T>::insert(T &data){
	if(root != NULL) return insert_help(this->root, data);

	// Create new root here
	Node<T> *new_root = new Node<T>(data);
	new_root->priority = rand();
	this->root = new_root;

	return false;
}

template <class T>
bool Treap<T>::insert_help(Node<T> *root, T &data){
	std::cout << "Insert help (" << root->data << "," << root->priority << ")" << std::endl;

	if(data <= root->data){
		// Goes to left side
		std::cout << "Going left.." << std::endl;
		if(root->left != NULL){
			// Sub-tree still exists
			return insert_help(root->left, data);
		}else{
			// Left tree is empty, new node there
			Node<T> *new_node = new Node<T>(data);
			new_node->priority = rand();
			root->left = new_node;
		}
	}else{
		// Goes to right side
		std::cout << "Going right.." << std::endl;
		if(root->right != NULL){
			// Sub-tree still exists
			return insert_help(root->right, data);
		}else{
			// Left tree is empty, new node there
			Node<T> *new_node = new Node<T>(data);
			new_node->priority = rand();
			root->right = new_node;
		}
	}

	return true;
}

template <class T>
void Treap<T>::dump() const{
	std::cout << "Test dump" << std::endl;
	if(root == NULL) return;

	if(this->root->left != NULL) dump_help(this->root->left);
	std::cout << " " << this->root->data;
	if(this->root->right != NULL) dump_help(this->root->right);
}
template <class T>
void Treap<T>::dump_help(Node<T> *root) const{
	if(root == NULL) return;

	if(root->left != NULL) dump_help(root->left);
	std::cout << " " << root->data;
	if(root->right != NULL) dump_help(root->right);
}


#endif
