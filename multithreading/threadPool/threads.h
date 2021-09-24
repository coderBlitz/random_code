#ifndef THREADS_H
#define THREADS_H

#include<condition_variable>
#include<functional>
#include<mutex>
#include<queue>
#include<thread>
#include<vector>


/***
	The type given defines the type of the argument given to the function.
	If no argument is desired, use type <void *> and always give nullptr. Or <int> and 0.
	TODO: Make template type the function definition (if multiargs/varargs possible)
***/
template <class T>
class ThreadPool{
public:
	ThreadPool(void);
	ThreadPool(unsigned int N);
	~ThreadPool(void);

	bool start(void);
	bool stop(void);
	bool push(std::function<void(int, T)> fn, T arg);
	void finish(void); // Wait till queue is empty

	unsigned int size(void){ return this->nthreads; }
	unsigned int length(void); // Return queue length

private:
	void toInfinityAndBeyond(int tid);

	bool running = false;
	bool stopping = false;
	std::mutex queue_mtx;
	std::condition_variable cv;
	std::queue<std::function<void(int, T)>> queue;
	std::queue<T> args;
	std::vector<std::thread> tds;
	unsigned int nthreads;
};


#include"threads.cpp"
#endif
