#ifndef TREAP_H
#define TREAP_H

template <class T>
class Node{
public:
	Node<T>(){
		this->left = NULL;
		this->right = NULL;
		this->priority = 0;
	}
	Node<T>(T &data){
		this->data = data;
		this->left = NULL;
		this->right = NULL;
		this->priority = 0;
	}

	T data;
	unsigned int priority;
	Node *left;
	Node *right;
};

template <class T>
class Treap{
public:
	Treap<T>();
	~Treap<T>();

	bool insert(T &data);

	void dump() const;
private:
	Node<T> *root;
	size_t size;

	void delete_help(Node<T> *root);
	bool insert_help(Node<T> *root,T &data);
	void dump_help(Node<T> *root) const;
};


#include"treap.cpp"
#endif
