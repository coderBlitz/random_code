/*	TODO: Verify data integrity
	TODO: Validate heap structure (currently implied from correct remove order)
	TODO: Verify update works
*/

#include<stdio.h>
#include<time.h>
#include"heap.h"

int N = 5000;
int *data;
int count;
int (*pri)(void *, void *);

int insert(int val){
	int i = count;
	while(i > 0 && pri((void *)(long)data[i-1], (void *)(long)val)){
		data[i] = data[i-1];

		--i;
	}

	data[i] = val;
	++count;
}

int main(){
	srand(time(0));

	data = malloc(N * sizeof(*data));
	if(data == NULL) return -1;
	count = 0;

	pri = HEAP_MAX;
	struct heap hp;
	heap_init(&hp, N/2, pri);

	printf("Inserting\n");
	for(int i = 0;i < N;++i){
		int val = rand();
		heap_insert(&hp, (void *)(long)val);
		insert(val);
		//heap_dump(&hp);
	}

	//for(int i = 0;i < count;++i) printf("data[%3d]: 0x%08X\n", i, data[i]);

	printf("Emptying\n");
	void * res = heap_pop(&hp);
	void * last;
	while(hp.size > 0){
		//heap_dump(&hp);
		last = res;
		res = heap_pop(&hp);
		if(pri(res, last) || res != (void *)(long)data[hp.size]){
			printf("Wrong!!!\tres: 0x%0lX\tlast: 0x%0lX\tdata: 0x%0lX\n", res, last, data[hp.size]);
			heap_dump(&hp);
			break;
		}
	}

	heap_destroy(&hp);
	free(data);
}
