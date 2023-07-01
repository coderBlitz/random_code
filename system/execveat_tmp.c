#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
//#include <linux/fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/sendfile.h>
#include <unistd.h>

int main() {
	// Create temp file.
	int tmp_fd = open("/tmp", O_WRONLY | O_TMPFILE | O_EXCL, S_IRWXU);
	if (tmp_fd == -1) {
		fprintf(stderr, "Opening tmpfile failed: %s\n", strerror(errno));
		return 1;
	}

	// Open file to copy
	int exec_fd = open("/bin/sleep", O_RDONLY | O_CLOEXEC);
	if (exec_fd == -1) {
		fprintf(stderr, "Opening exec file failed: %s\n", strerror(errno));
		return 2;
	}

	// Copy file
	char buffer[8192];
	int res;
	off_t offset = 0;
	while ((res = sendfile(tmp_fd, exec_fd, &offset, 8192)) > 0) {}

	printf("My pid: %d\n", getpid());
	sleep(5);

	char *args[] = {"Bla", "10", NULL};
	char *env[] = {NULL};
	res = execveat(exec_fd, "", args, env, AT_EMPTY_PATH);

	fprintf(stderr, "Execveat FD failed: %s\n", strerror(errno));

	char path[64];
	snprintf(path, 64, "/proc/self/fd/%d", tmp_fd);
	res = execveat(-1, path, args, env, 0);

	fprintf(stderr, "Execveat proc failed: %s\n", strerror(errno));

	close(exec_fd);
	close(tmp_fd);

	return 0;
}
