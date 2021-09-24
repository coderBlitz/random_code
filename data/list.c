/*	list.c -- An odd growing array implementation
Notes:
	Originated as a misinterpretation of an old 341 project (before I took it).
	Dynamically allocates new arrays of twice the original size, when needed.
	 Stores the arrays in an array. Visually, this would be a logarithmic
	 triangular matrix.
*/
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>

const int init_size = 16;// Init size of number of rows (gives us roughly 2^init_size spaces)

struct list{
	unsigned long row_cap;// Initial size of first row (successive rows twice the previous; user defined)
	unsigned long size;// Actual amount of data currently stored
	long **buffer;// The buffer which holds the rows and data
};

void addData(struct list *buf, long data){
	if(buf->buffer == NULL && buf->row_cap > 0){ // If whole structure is not allocated
//		printf("Creating buffer..\n");
		buf->buffer = malloc(init_size*sizeof(long *));
		if(buf->buffer == NULL){
			printf("Initial structure allocate failed.\n");
			exit(1);
		}

		buf->buffer[0] = malloc(buf->row_cap * sizeof(long));
		if(buf->buffer[0] == NULL){
			printf("Initial row allocation failed.\n");
			exit(1);
		}

		for(int i = 1;i < init_size;i++) buf->buffer[i] = NULL;

		buf->buffer[0][0] = data;
//		printf("Data at (0,0) = %ld\tcurSize = 0\n",data);
		buf->size = 1;
	}else if(buf->buffer != NULL){ // If the structure is allocated (all but first time)
		// Log base-2 because double size each time, so row_size*2^n
//		unsigned long row = log(buf->size/buf->row_cap + 1)/log(2); // Should work same as below
		unsigned long row = floor(log(buf->size/buf->row_cap + 1)/log(2)); // Floor adds consistency

//		printf("Test row = %d\n",row);

		long pos = (row == 0)?buf->size:buf->size - buf->row_cap*(int)pow(2,row) + buf->row_cap;

		if(buf->buffer[row] != NULL) buf->buffer[row][pos] = data;
		else{
			unsigned long new_size = buf->row_cap*(int)pow(2.0,row);
			printf("Allocating row %d with size %d\n",row,new_size);

			buf->buffer[row] = malloc(new_size * sizeof(long));
			if(buf->buffer[row] == NULL){
				printf("Failed to allocate row %d\n",row);
				exit(1);
			}

			buf->buffer[row][pos] = data;
		}
//		printf("Data at (%d,%ld) = %ld\tcurSize = %d\n\n",row,pos,data,buf->size);
		buf->size++;
	}
}

long at(struct list *buf, unsigned long index){
	unsigned long row = floor(log(index/buf->row_cap + 1)/log(2)); // Floor adds consistency

	long pos = (row == 0)?index:index - buf->row_cap*(int)pow(2,row) + buf->row_cap;

	return buf->buffer[row][pos];
}

void print(struct list *buf){
	if(buf->size > 0){
		unsigned long total = 0;
		unsigned long count = 1;
		for(int i = 0;i < init_size;i++){
//			printf("Row: %d\n",i);
			long numEntries = (i == 0)?buf->row_cap:buf->row_cap*(count);
			if(buf->buffer[i] != NULL){
//				printf("numEntries = %ld\n",numEntries);
				for(int j = 0;j < numEntries;j++){
//					printf("total = %d v curSize = %d\n",total,buf->size);
					if(++total > buf->size) break;
					printf("(%d,%d): %lu\n",i,j,buf->buffer[i][j]);
				}
//				sleep(1);
			}
			count *= 2;
		}
	}
}

void destroy(struct list *buf){
	if(buf->buffer != NULL){
		for(int i = 0;i < init_size;i++){
			if(buf->buffer[i] != NULL){
				printf("Dealloc Row: %d\n",i);
				free(buf->buffer[i]);
			}
		}
		free(buf->buffer);// Delete last part
	}
	buf->row_cap = 0;
	buf->size = 0;
}

size_t getCapacity(const struct list *buf){
	size_t total = 0;
	for(int i = 0;i < init_size;i++){
		if(buf->buffer[i] == NULL) break;
		total += buf->row_cap*pow(2,i);
//		printf("Total now %lu\n",total);
	}

	return total;
}

int main(){
	struct list c;
	c.row_cap = 3;
	c.buffer = NULL;

	for(int i = 0;i < 50;i++){
		addData(&c, i+1);
	}

//	for(int i = 0;i < 10;i++) printf("list[%d]: %lu\n", i, at(&c, i));

	print(&c);
	printf("Total storage: %lu\nCurrent size: %lu\n",getCapacity(&c), c.size);
	destroy(&c);
}
