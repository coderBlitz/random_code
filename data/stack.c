#include<stdio.h>
#include"stack.h"

static void **data_stack;
static size_t stack_capacity = 0xFF;
static size_t stack_size = 0;

int push(void *ptr){
	if(data_stack == NULL){
		data_stack = malloc(stack_capacity * sizeof(*data_stack));
		if(data_stack == NULL){
			printf("Could not allocate stack\n");
			return -1;
		}
	}

	if(stack_size < stack_capacity){
		//printf("Pushing %p\n", ptr);
		data_stack[stack_size++] = ptr;
	}else{
	//if(stack_size == stack_capacity){
		stack_capacity *= 2;
		void **temp = realloc(data_stack, stack_capacity * sizeof(*data_stack));
		if(temp == NULL){
			fprintf(stderr, "Could not expand stack size to %lu\n", stack_capacity);
			stack_capacity >>= 2;
			return -1;
		}
		data_stack = temp;
	}
	
	return 0;
}

void *pop(void){
	if(stack_size == 0) return NULL;

	stack_size--;
	void *rv = data_stack[stack_size];
	data_stack[stack_size] = NULL;

	if(stack_size == 0){
		free(data_stack);
		data_stack = NULL;
	}

	return rv;
}

size_t stack_count(){
	return stack_size;
}
