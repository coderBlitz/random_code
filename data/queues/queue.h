#ifndef QUEUE_H_
#define QUEUE_H_

#include<linux/errno.h>
#include<linux/kernel.h>

#ifndef QUEUE_DATA_TYPE
	#define QUEUE_DATA_TYPE long
#endif

typedef QUEUE_DATA_TYPE queue_data_t;

typedef struct queue_node{
	queue_data_t data;
	struct queue_node* next;
	struct queue_node* tail;
} queue_node;

long queueAppend(queue_node**, queue_data_t);
long queueRemove(queue_node**);

#include"queue.c"

#endif
