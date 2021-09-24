#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/resource.h>

/*
Whether or not certain of these fields display valid information is governed by
 a ptrace access mode PTRACE_MODE_READ_FSCREDS | PTRACE_MODE_NOAUDIT check
 (refer to ptrace(2)). If the check denies access, then the field value is
 displayed as 0. The affected fields are indicated with the marking [PT].
*/
struct task_stat {
	int pid;
	char comm[16];
	char state;
	int ppid;
	int pgrp;
	int session;
	int tty_nr;
	int tpgid;
	unsigned flags;
	unsigned long minflt;
	unsigned long cminflt;
	unsigned long majflt;
	unsigned long cmajflt;
	unsigned long utime;
	unsigned long stime;
	long cutime;
	long cstime;
	long priority;
	long nice;
	long num_threads;
	long itrealvalue;
	unsigned long long starttime;
	unsigned long vsize;
	long rss;
	unsigned long rsslim;

	unsigned long startcode; // PT
	unsigned long endcode; // PT
	unsigned long startstack; // PT
	unsigned long kstkesp; // PT
	unsigned long kstkeip; // PT

	unsigned long signal;
	unsigned long blocked;
	unsigned long sigignore;
	unsigned long sigcatch;

	unsigned long wchan; // PT

	unsigned long nswap;
	unsigned long cnswap;
	int exit_signal;
	int processor;
	unsigned rt_priority;
	unsigned policy;
	unsigned long long delayacct_blkio_ticks;
	unsigned long guest_time;
	unsigned long cguest_time;

	unsigned long start_data; // PT
	unsigned long end_data; // PT
	unsigned long start_brk; // PT
	unsigned long arg_start; // PT
	unsigned long arg_end; // PT
	unsigned long env_start; // PT
	unsigned long env_end; // PT
	int exit_code; // PT
};

int main(){
	struct rusage rsrc;

	int res = getrusage(RUSAGE_SELF, &rsrc);
	if(res) return -1;

	printf("MaxRSS: %lu\n", rsrc.ru_maxrss);
	printf("ixRSS: %lu\n", rsrc.ru_ixrss);
	printf("idRSS: %lu\n", rsrc.ru_idrss);
	printf("isRSS: %lu\n", rsrc.ru_isrss);
	printf("MinFLT: %lu\n", rsrc.ru_minflt);
	printf("MajFLT: %lu\n", rsrc.ru_majflt);
	printf("NSwap: %lu\n", rsrc.ru_nswap);
	printf("inblock: %lu\n", rsrc.ru_inblock);
	printf("oublock: %lu\n", rsrc.ru_oublock);
	printf("msgsnd: %lu\n", rsrc.ru_msgsnd);
	printf("msgrcv: %lu\n", rsrc.ru_msgrcv);
	printf("nsignals: %lu\n", rsrc.ru_nsignals);
	printf("nvcsw: %lu\n", rsrc.ru_nvcsw);
	printf("nivcsw: %lu\n", rsrc.ru_nivcsw);

	/* Read proc/[pid]/status file
	*/
	printf("\n");
	char buffer[4096];
	FILE *fp = NULL;
	fp = fopen("/proc/self/statm", "r");
	if(fp == NULL) return -1;

	char buffer[4096];
	buffer[128] = 0;
	while(fread(buffer, 1, 128, fp) != 0){
		printf("%s", buffer);
	}
	printf("\n");

	fclose(fp);

	/* Parse/tokenize proc/[pid]/stat
	*/
	printf("\n");
	fp = fopen("/proc/self/stat", "r");
	if(fp == NULL) return -1;

	size_t ret = fread(buffer, 1, 4095, fp);
	buffer[ret] = 0;
	printf("%s\n", buffer);

	fclose(fp);

	// Search for opening parenthesis of command name
	char *next = strchr(buffer, '(');
	next += 16; // Command length limited to TASK_COMM_LEN (presently 16, so only 15 chars print)
	while(*next != ')') --next; // Find closing parenthesis

	char *c_tmp = buffer;
	struct task_stat ts;
	ts.pid = strtol(buffer, &c_tmp, 10);
	printf("pid: %d\n", ts.pid);

	c_tmp += 2; // Skip space and opening parenthesis
	memcpy(ts.comm, c_tmp, next - c_tmp);
	ts.comm[15] = 0;
	printf("comm: %s\n", ts.comm);

	c_tmp = next + 2; // Skip to start of next field
	ts.state = *c_tmp;
	printf("state: %c\n", ts.state);

	c_tmp += 2; // Next field
	ts.ppid = strtol(c_tmp, &c_tmp, 10);
	printf("ppid: %d\n", ts.ppid);

	ts.pgrp = strtol(c_tmp, &c_tmp, 10);
	printf("pgrp: %d\n", ts.pgrp);

	ts.session = strtol(c_tmp, &c_tmp, 10);
	printf("session: %d\n", ts.session);

	return 0;
}
