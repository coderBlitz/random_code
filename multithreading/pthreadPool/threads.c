#include"threads.h"

/**	Initializes pool struct with nthreads, or gets available processor count if
	 nthreads = 0.
	Returns num_threads.
**/
int pool_init(struct thread_pool *p, unsigned short nthreads){
	p->num_jobs = 0;
	p->jobs = NULL;
	p->last_job = NULL;
	p->RUNNING = 0;
	p->STOPPING = 0;

	// Set thread count, or get and set.
	if(nthreads){
		p->num_threads = nthreads;
		//printf("%hd threads requested\n", nthreads);
	}else{
		long np = sysconf(_SC_NPROCESSORS_ONLN);
		if(np == -1){
			fprintf(stderr, "Could not get online processors.\n");
			np = 1;
		}
		//printf("%ld processors available.\n", np);
		p->num_threads = np;
	}

	// Allocate thread ID array
	p->threads = malloc(p->num_threads * sizeof(*p->threads));
	if(p->threads == NULL){
		fprintf(stderr, "Could not alloc thread ID array.\n");
		p->num_threads = 0;
		return 0;
	}

	// Init pthread stuff
	pthread_mutex_init(&p->qlock, NULL);
	pthread_cond_init(&p->cond, NULL);

	return p->num_threads;
}

/**	Stop threads and zero/clear everything
**/
void pool_destroy(struct thread_pool *p){
	// Stop threads
	pool_stop(p, 1);

	// Delete all jobs
	struct pool_job *ptr1 = p->jobs, *ptr2;
	while(ptr1){
		ptr2 = ptr1->next;
		free(ptr1);
		ptr1 = ptr2;
	}

	// Clear everything
	p->num_jobs = 0;
	p->jobs = NULL;
	p->last_job = NULL;
	p->RUNNING = 0;
	p->STOPPING = 0;
	p->num_threads = 0;
	free(p->threads);
	p->threads = NULL;
	pthread_mutex_destroy(&p->qlock);
	pthread_cond_destroy(&p->cond);
}

/**	Adds job to back of queue
**/
int pool_push(struct thread_pool *p, pool_fn fn, void *args){
	//printf("Pushing job..\n");

	struct pool_job *new_job = malloc(sizeof(*new_job));
	if(new_job == NULL){
		return -1;
	}
	new_job->fn = fn;
	new_job->args = args;
	new_job->next = NULL;

	// Lock and insert job to queue tail
	pthread_mutex_lock(&p->qlock);
	p->num_jobs += 1;
	if(p->jobs == NULL){
		p->jobs = new_job;
		p->last_job = new_job;
	}else{
		p->last_job->next = new_job;
		p->last_job = new_job;
	}
	pthread_mutex_unlock(&p->qlock);

	// Signal a thread to wake up (if waiting) and take job
	pthread_cond_signal(&p->cond);

	return 0;
}

/**	Clear queue
**/
int pool_clear(struct thread_pool *p){
	// Delete all jobs
	struct pool_job *ptr1, *ptr2;

	// Remove queue references and zero count
	pthread_mutex_lock(&p->qlock);
	ptr1 = p->jobs;
	p->jobs = NULL;
	p->last_job = NULL;
	p->num_jobs = 0;
	pthread_mutex_unlock(&p->qlock);

	while(ptr1){
		ptr2 = ptr1->next;
		free(ptr1);
		ptr1 = ptr2;
	}
}

/**	The worker function that every thread will fun
**/
void *pool_thread_loop(void *pool_arg){
	struct thread_pool *const p = (struct thread_pool *)pool_arg;
	struct pool_job *job = NULL;

	// Assign 0-based thread IDs, which get passed to the jobs
	const pthread_t self = pthread_self();
	int tid;
	for(int i = 0;i < p->num_threads;i++){
		if(p->threads[i] == self){
			tid = i;
			break;
		}
	}

	// Variables for each job
	pool_fn fn;
	void *args = NULL;

	//printf("Thread %d starting..\n", tid);
	while(!p->STOPPING){
		// Lock and wait on condition
		pthread_mutex_lock(&p->qlock); /** START CRITICAL **/
		while(p->num_jobs == 0 && !p->STOPPING){
			pthread_cond_wait(&p->cond, &p->qlock);
		}

		// Check conditions
		if(p->num_jobs == 0 || p->STOPPING){
			pthread_mutex_unlock(&p->qlock);
			continue;
		}

		// Remove node from queue, set  next pointers
		job = p->jobs;
		p->num_jobs -= 1;
		p->jobs = p->jobs->next;
		if(p->num_jobs == 0){
			p->last_job = NULL;
		}
		pthread_mutex_unlock(&p->qlock); /** END CRITICAL **/

		// Get function and args
		fn = job->fn;
		args = job->args;

		free(job); // Free job node

		// Call job
		//printf("calling job..\n");
		fn(tid, args);
	}
	//printf("Thread %d stopping..\n", tid);

	return NULL;
}

int pool_start(struct thread_pool *p){
	if(p->RUNNING) return 1;
	p->STOPPING = 0;

	int res;
	for(int i = 0;i < p->num_threads;i++){
		//printf("Launching thread %d..\n", i);
		res = pthread_create(p->threads + i, PTHREAD_CANCEL_ENABLE | PTHREAD_CANCEL_DEFERRED, pool_thread_loop, p);
		if(res){
			fprintf(stderr, "Error launching %d: %d\n", i, res);
		}
	}

	p->RUNNING = 1;
}

/**	Join all threads.
**/
int pool_stop(struct thread_pool *p, const char force){
	if(!p->RUNNING) return 1;
	p->STOPPING = 1;

	pthread_cond_broadcast(&p->cond);
	for(int i = 0;i < p->num_threads;i++){
		if(force){
			//printf("Cancelling thread %d..\n", i);
			pthread_cancel(p->threads[i]);
		}
		//printf("Joining %d..\n", i);
		pthread_join(p->threads[i], NULL);
	}
	//printf("All threads stopped!\n");

	p->STOPPING = 0;
	p->RUNNING = 0;
}

void pool_finish(struct thread_pool *p, long interval){
	//printf("Waiting for queue to empty..\n");

	struct timespec t = {0, (interval) ? interval : wait_delay_nsec};
	while(p->num_jobs){
		nanosleep(&t, NULL);
	}
}
