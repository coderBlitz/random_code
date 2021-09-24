#ifndef THREADS_H
#define THREADS_H

#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<unistd.h>

typedef void (*pool_fn)(int, void *);

static const long wait_delay_nsec = 100000; // 100 usec == 0.1 msec

struct pool_job{
	pool_fn fn;
	void *args;
	struct pool_job *next;
};

struct thread_pool{
	pthread_mutex_t qlock;
	pthread_cond_t cond;
	size_t num_jobs;
	struct pool_job *jobs;
	struct pool_job *last_job;
	char RUNNING;
	char STOPPING;
	unsigned short num_threads;
	pthread_t *threads;
};

/**	Initializes pool struct with nthreads, or gets available processor count if
	 nthreads = 0.
**/
int pool_init(struct thread_pool *p, unsigned short nthreads);

/**	Stop threads, empty the queue, zero all members, and free all resources.
**/
void pool_destroy(struct thread_pool *p);

/**	Add a job to the queue, to call fn with given arg.
**/
int pool_push(struct thread_pool *p, pool_fn fn, void *args);

/**	Clear the job queue
**/
int pool_clear(struct thread_pool *p);

/**	Start and stop all threads
**/
int pool_start(struct thread_pool *p);
int pool_stop(struct thread_pool *p, const char force);

/**	Wait for queue to empty, sleeping for interval if non-zero, default otherwise.
**/
void pool_finish(struct thread_pool *p, long interval);

#endif
