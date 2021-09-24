#include<iostream>
#include<sstream>
#include<string>
//#include"dtree.h"

using std::cout;
using std::endl;
using std::string;

void printTree(string tree, const unsigned int degree = 0){
	if(tree == "") return;

	// Remove outermost parenthesis
	string cur = tree.substr(1, tree.length() - 2);
	string left = "";
	string right = "";

	size_t step = 0;
	size_t pos = 0;

	// Parse left segment, if exists
	if(cur[0] == '('){
		// Find matching parenthesis
		do{
			if(cur[pos] == '(') step++;
			else if(cur[pos] == ')') step--;

			++pos;
		}while(step > 0 && pos < cur.length());

		// Grab the left segment
		left = cur.substr(0, pos);
		cur.erase(0, pos); // Remove left from cur
		step = 0;
		pos = 0;
		//cout << "Left: " << left << endl;
	}

	// Find start of next opening parenthesis, for right side
	while(cur[pos++] != '(' && pos < cur.length()){}

	// Grab right substring, if it exists
	if(cur[pos-1] == '('){
		right = cur.substr(pos - 1);
		cur.erase(pos - 1);
		//cout << "Right: " << right << endl;
	}

	// Parse (optionally)
	/*size_t pos = cur.find(':');
	int disc = cur.substr(0, pos);
	cur.erase(0, pos + 1);

	pos = cur.find(':');
	int size = cur.substr(0, pos);
	cur.erase(0, pos + 1);

	pos = cur.find(':');
	int numVacant = cur.substr(0, pos);
	cur.erase(0, pos + 1);*/

	// Print (change indent at leisure)
	string indent = "  ";
	for(int i = 0;i < degree;i++) cout << indent;
	cout << cur << endl;

	// Recurse print
	printTree(left, degree + 1);
	printTree(right, degree + 1);	
}

/*void pprintDT(DTree &dt){
	std::stringstream str;

	// Save and redirect stream
	std::streambuf *coutbuf = cout.rdbuf(str.rdbuf());

	// Call dump
	dt.dump();

	// Reset stream
	cout.rdbuf(coutbuf);

	// Call print
	printTree(str.str());
}*/

int main(int argc, char *argv[]){
	//string str = "((((207:1:0)251:5:0(883:3:0((1503:1:0)1691:2:0)))1980:10:0((2247:1:0)2636:4:0(2655:2:0(2987:1:0))))4430:20:0(((4945:1:0)4985:4:0(5833:2:0(6336:1:0)))7092:9:0((7488:1:0)7605:4:0(7713:2:0(8319:1:0)))))";
	//string str = "((1503:1:0)1691:2:0(1234:1:0))";
	string str = "((((121:1:0)341:4:0(543:2:0(656:1:0)))789:9:0((987:1:0)988:4:0(989:2:0(1254:1:0))))1567:22:0(((1864:1:0)1896:6:0(2000:4:0(4562:3:0(6009:2:0(6100:1:0)))))6123:12:0((6300:1:0)6578:5:0(7345:3:0(8000:2:0(8907:1:0))))))";

	printTree(str);

	/*
	DTree dt;
	// Do insertions
	pprintDT(dt);
	*/

	return 0;
}
