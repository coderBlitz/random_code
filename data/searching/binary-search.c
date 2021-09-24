#include<stdio.h>
#include<stdlib.h>

int main(){
	int capacity = 9;
	int size = capacity;
	int m_start = 0;
	int data = 10;

	int *m_data = calloc(sizeof(int), capacity);
	for(int i=0;i<size;i++) m_data[i] = i+1;

	int search_size = size;
	int mid = (search_size/2 + m_start)%capacity;// Size/2 gives middle value, m_start gives offset, modulo capacity allows wrap

	while(search_size != 0 && mid > 0 && mid < size){
		search_size /= 2;
		printf("search_size = %d\tmid = %d\n", search_size, mid);
		if(m_data[mid] == data){ printf("Found %d at position %d\n", data, mid); break; }
		else if(m_data[mid] < data) mid += (search_size)%capacity;
		else mid -= (search_size)%capacity;
//		if(search_size == 0 || mid < 0 || mid > size) break;
	}

	free(m_data);
} 
