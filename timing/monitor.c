#include<errno.h>
#include<signal.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/mman.h>
#include<sys/types.h>
#include<sys/wait.h>
#include<time.h>
#include<unistd.h>

#define USEC_NANO 1000
#define MSEC_NANO 1000000
#define SEC_NANO 1000000000
#define MIN_NANO 60000000000

void timespecsub(struct timespec *res, const struct timespec *a, const struct timespec *b){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

unsigned long time_scale(char c){
	if(c == 'S' || c == 's'){
		return SEC_NANO;
	}else if(c == 'M'){
		return MIN_NANO;
	}else if(c == 'm'){
		return MSEC_NANO;
	}else if(c == 'U' || c == 'u'){
		return USEC_NANO;
	}

	return 1;
}

int main(int argc, char *argv[]){
	if(argc < 2){
		printf("Usage: ./monitor [options] program\n");
		printf("\tOptions:\n");
		printf("\t\t-i INT\tInterval (nsec) to probe process. Accepts suffixes (n,u,m,s,M).\n");
		printf("\t\t-o FILE\tOutput file for recorded data.\n");

		return 1;
	}

	// Runtime parameters
	struct timespec interval = {0, 5 * MSEC_NANO};
	char *OUT_FILE = "out.csv";

	// Parse args
	int arg_start = -1;
	for(int i = 1;i < argc;i++){
		if(argv[i][0] != '-'){
			arg_start = i;
			break;
		}

		unsigned long tmp, scale;
		char *check;
		switch(argv[i][1]){
			// Nanosleep interval
			case 'i':
				tmp = strtoul(argv[++i], &check, 10);
				if(tmp != 0){
					scale = time_scale(check[0]);
					tmp *= scale;
					interval.tv_sec = tmp / SEC_NANO;
					interval.tv_nsec = tmp % SEC_NANO;
				}
				break;
			case 'o':
				OUT_FILE = argv[++i];
				break;
			default:
				printf("Unknown flag '%c'\n", argv[i][1]);
		}
	}

	if(arg_start < 0){
		fprintf(stderr, "No executable given.\n");
		return 1;
	}

	// Create shared mem to start timer (only enough to pass a flag)
	size_t map_len = 1;
	char *notif = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if(notif == NULL){
		perror("Could not share memory");
		return 1;
	}
	notif[0] = 0; // Just in case MAP_ANONYMOUS does not initialize

	printf("Interval: %lu.%09ld\n", interval.tv_sec, interval.tv_nsec);
	printf("Output: '%s'\n", OUT_FILE);

	struct timespec start, end, diff, ts1;

	// Fork and exec child
	pid_t pid = fork();
	if(pid <  0){
		perror("Fork failed");
		return 1;
	}else if(!pid){
		/* Child
		*/
		while(notif[0] != 1){}

		execvp(argv[arg_start], argv + arg_start);
		perror("Failed to execute");

		munmap(notif, map_len);
		exit(1);
	}

	/* Parent
	*/
	// Open the proc file before loop
	printf("pid: %lu\n", pid);

	char stat_path[32];
	snprintf(stat_path, 32, "/proc/%u/statm", pid);

	FILE *fp = fopen(stat_path, "r");
	if(fp == NULL){
		kill(pid, SIGTERM);
		return 2;
	}

	/* Do runtime analysis
	*/
	int status = 0, count = 0, field = 2; // Field 2 grows with usage, field 6 grows with allocation
	ssize_t ret;
	pid_t cid = 0;
	char buffer[256];
	char *walk;
	void *p_tmp;

	// Vars to use for probes
	size_t entry_cap = 4096;
	size_t entry_cnt = 0;

	unsigned long *records = malloc(entry_cap * sizeof(*records));
	if(records == NULL){
		perror("Malloc failed");
		kill(pid, SIGTERM);
		fclose(fp);

		return 2;
	}

	// Vars for time records
	double *times = malloc(entry_cap * sizeof(*times));
	if(times == NULL){
		perror("Malloc failed");
		kill(pid, SIGTERM);
		fclose(fp);
		free(records);

		return 2;
	}

	// Notify child to start
	notif[0] = 1;
	munmap(notif, map_len); // Clear map, since unneeded below

	// Start time
	clock_gettime(CLOCK_MONOTONIC, &start);

	do{
		nanosleep(&interval, NULL);

		// Read proc then rewind for next read
		ret = fread(buffer, 1, 255, fp);
		buffer[ret] = 0;
		rewind(fp);

		// Find field
		walk = buffer;
		count = 1;
		while(count != field){
			if(*(++walk) == ' ') count++;
		}

		// Record parsed value
		records[entry_cnt] = strtoul(walk, NULL, 10);

		// Record time
		clock_gettime(CLOCK_MONOTONIC, &ts1);
		timespecsub(&diff, &ts1, &start);
		times[entry_cnt] = diff.tv_sec + diff.tv_nsec / (double) SEC_NANO;
		//printf("%lu\n", mem);

		// Expand arrays when necessary
		entry_cnt += 1;
		if(entry_cnt >= entry_cap){
			entry_cap *= 2;

			// Record realloc
			p_tmp = realloc(records, entry_cap * sizeof(*records));
			if(p_tmp == NULL){
				perror("Realloc failed");
				kill(pid, SIGTERM);
				fclose(fp);
				free(records);
				free(times);

				return 2;
			}
			records = p_tmp;

			// Time realloc
			p_tmp = realloc(times, entry_cap * sizeof(*times));
			if(p_tmp == NULL){
				perror("Realloc failed");
				kill(pid, SIGTERM);
				fclose(fp);
				free(records);
				free(times);

				return 2;
			}
			times = p_tmp;
		}

		// Iterate stuff
		cid = waitpid(pid, &status, WNOHANG);
	}while(cid == 0 || (!WIFEXITED(status) && !WIFSIGNALED(status)));

	// Get end time
	clock_gettime(CLOCK_MONOTONIC, &end);

	timespecsub(&diff, &end, &start);

	// Do some maintenance stuff
	fclose(fp);

	if(entry_cnt) entry_cnt -= 1; // Last result is often 0, so just ignore.

	//printf("start: %lu.%09ld\n", start.tv_sec, start.tv_nsec);
	//printf("end: %lu.%09ld\n", end.tv_sec, end.tv_nsec);
	printf("time: %lu.%09ld\n", diff.tv_sec, diff.tv_nsec);

	/* Output/save results
	*/
	printf("%lu probes\n", entry_cnt);

	fp = fopen(OUT_FILE, "w");
	if(fp == NULL){
		perror("Cannot open output file");
		free(records);

		return 3;
	}

	fprintf(fp, "elapsed time,memory\n");
	for(size_t i = 0;i < entry_cnt;i++){
		fprintf(fp, "%.9lf,%lu\n", times[i], records[i]);
	}

	fclose(fp);

	free(records);
	free(times);

	return 0;
}
