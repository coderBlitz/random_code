#ifndef QUEUE_C_
#define QUEUE_C_

#include"queue.h"

long queueAppend(queue_node** queue, queue_data_t data){
	queue_node *new_node;

	// Cannot create a queue from nothing
	if(queue == NULL) return -EFAULT;

	new_node = kmalloc(sizeof(*new_node), GFP_KERNEL);
	if(new_node == NULL){
		return -ENOMEM;
	}

	new_node->data = data;
	new_node->next = NULL;

	if(*queue == NULL){
		(*queue) = new_node;
	}else{
		(*queue)->tail->next = new_node;
	}
	(*queue)->tail = new_node;

	return 0;
}

long queueRemove(queue_node **queue){
	queue_node *cur, *tail, *next;

	if(queue == NULL || *queue == NULL) return -EFAULT;

	cur = *queue;
	next = (*queue)->next;
	tail = (*queue)->tail;

	(*queue) = next;
	if(next != NULL){
		next->tail = tail;
	}

	// Delete after here
	kfree(cur);

	return 0;
}

#endif
