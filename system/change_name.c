/*
Note:
	From postgresql source, if you save the environment elsewhere, you can use
	 addresses from the start of argv[0] to end of environment strings.
	May be able to modify argv table and envp table pointers, but may not have
	 any effect.
*/

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

int main(int argc, char **argv) {
	printf("Old argv[0] is %s\n", argv[0]);

	// Set new argv name (sets /proc/self/cmdline; what shows up in ps -f)
	char *new_name = "argv";
	int max_len = strlen(argv[0]);
	//strncpy(argv[0], new_name, max_len);
	char *dst = argv[0] + max_len + 1;
	int orig_1 = strlen(argv[1]);
	char *pt = argv[1];
	strncpy(pt, "My string", orig_1+1); // Really doesn't like working if writing beyond original length
	pt += orig_1-2;
	*pt = 0;
	//*(argv + 1 + orig_1 + 1) = 0;
	//dst += strlen("My string") + 1;
	//*dst = 0;
	printf("New argv[0] is %s\n", argv[0]);
	printf("New argv[1] is %s\n", argv[1]);

	// Write to /proc/self/comm
	/*int tid = gettid();
	char comm_path[64];
	snprintf(comm_path, 64, "/proc/self/task/%d/comm", tid);
	printf("Comm path = '%s'\n", comm_path);
	FILE *fp = fopen(comm_path, "rb");
	fwrite(new_name, 1, strlen(new_name)+1, fp);
	fclose(fp);*/

	// prctl set name (sets /proc/self/comm and name in /proc/self/status)
	char *prctl_name = "prctl";
	int res = prctl(PR_SET_NAME, prctl_name, 0, 0, 0);
	if (res < 0) {
		fprintf(stderr, "pctrl set name failed: %d\n", res);
	}

	printf("PID is %d\n", getpid());
	sleep(12);

	return 0;
}
