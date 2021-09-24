#ifndef THREADS_CPP
#define THREADS_CPP

#include"threads.h"


template <class T>
ThreadPool<T>::ThreadPool(void){
	int rec = std::thread::hardware_concurrency();

	if(rec) this->nthreads = rec;
	else this->nthreads = 2; // Safe default
}
template <class T>
ThreadPool<T>::ThreadPool(unsigned int N){
	this->nthreads = N;
}
template <class T>
ThreadPool<T>::~ThreadPool(void){
	this->stop();
}


template <class T>
void ThreadPool<T>::toInfinityAndBeyond(int tid){
	std::function<void(int, T)> job;
	T arg;
	while(!stopping){
		{
			std::unique_lock<std::mutex> lck(this->queue_mtx);
			this->cv.wait(lck, [this]{return !this->queue.empty() || this->stopping;});
			if(stopping) break; // Don't touch queues when stopping

			if(!this->queue.empty()){
				job = this->queue.front();
				this->queue.pop();
			}
			if(!this->args.empty()){
				arg = this->args.front();
				this->args.pop();
			}
		}
		try{
			job(tid, arg);
		}catch(std::exception e){}
	}
}

/**	Launch all threads
**/
template <class T>
bool ThreadPool<T>::start(void){
	if(running) return false;

	for(int i = 0;i < this->nthreads;i++){
		tds.push_back(std::thread(&ThreadPool::toInfinityAndBeyond, this, i));
	}
	this->running = true;

	return true;
}

/**	Add a job to the queue
**/
template <class T>
bool ThreadPool<T>::push(std::function<void(int, T)> fn, T arg){
	//if(!running) return false; // Only if jobs should be refused when not running

	{
		std::unique_lock<std::mutex> lck(this->queue_mtx);
		this->queue.push(fn);
		this->args.push(arg);
	}
	this->cv.notify_one();

	return true;
}

/**	Stop all running threads (gracefully)
**/
template <class T>
bool ThreadPool<T>::stop(void){
	if(!running) return false;

	{
		std::unique_lock<std::mutex> lock(this->queue_mtx);
		this->stopping = true;
	}

	this->cv.notify_all();
	for(int i = 0;i < this->tds.size();i++){
		this->tds[i].join();
	}
	this->tds.clear();
	this->stopping = false;
	this->running = false;

	return true;
}

/**	Consider having the interval be specifiable
**/
template <class T>
void ThreadPool<T>::finish(void){
	do{
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}while(this->length() > 0);
}

template <class T>
unsigned int ThreadPool<T>::length(void){
	unsigned int ret;
	{
		std::unique_lock<std::mutex> lock(this->queue_mtx);
		ret = this->queue.size();
	}

	return ret;
}

#endif
