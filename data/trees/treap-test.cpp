#include<iostream>
#include"treap.h"

using std::cout;
using std::endl;

int main(){
	Treap<int> t;

	int a = 0;
	for(int i = 0;i < 3;++i){
		cout << "Inserting " << a << endl;
		a = 5 + i;
		t.insert(a);
		t.dump();
	}

	return 0;
}
