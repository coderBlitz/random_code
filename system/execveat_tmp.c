#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
//#include <linux/fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
	// Create temp file.
	/*int tmp_fd = open("/tmp", O_WRONLY | O_TMPFILE | O_EXCL, S_IRWXU);
	if (tmp_fd == -1) {
		fprintf(stderr, "Opening tmpfile failed: %s\n", strerror(errno));
		return 1;
	}*/
	int tmp_fd = memfd_create("tmpmem", 0);
	if (tmp_fd == -1) {
		fprintf(stderr, "Opening memfd failed: %s\n", strerror(errno));
		return 1;
	}

	// Open file to copy
	int exec_fd = open("/bin/sleep", O_RDONLY | O_CLOEXEC);
	if (exec_fd == -1) {
		fprintf(stderr, "Opening exec file failed: %s\n", strerror(errno));
		return 2;
	}

	// Proc path
	char proc_path[64];
	snprintf(proc_path, 64, "/proc/self/fd/%d", tmp_fd);

	// Copy file
	char buffer[8192];
	int res;
	off_t offset = 0;
	while ((res = sendfile(tmp_fd, exec_fd, &offset, 8192)) > 0) {}

	// Just in case perms
	/*res = fchmod(tmp_fd, S_IRWXU);
	if (res < 0) {
		fprintf(stderr, "fchmod failed");
	}
	res = chmod(proc_path, S_IRWXU);
	if (res < 0) {
		fprintf(stderr, "chmod failed");
	}*/

	// Sleep for noting PID
	printf("My pid: %d\n", getpid());
	sleep(5);

	char *args[] = {"Random process name", "10", NULL};
	char *env[] = {NULL};

	// Exec the tmp fd directly
	printf("Trying execveat on fd directly..\n");
	res = execveat(tmp_fd, "", args, env, AT_EMPTY_PATH);
	fprintf(stderr, "Execveat FD failed: %s\n", strerror(errno));

	// Exec the proc path directly
	printf("Trying /proc/self/fd/ execveat..\n");
	res = execveat(-1, proc_path, args, env, 0);
	fprintf(stderr, "Execveat proc failed: %s\n", strerror(errno));

	// Try execve normally just in case
	printf("Trying /proc/self/fd/ execve..\n");
	res = execve(proc_path, args, env);
	printf("Execve failed\n");

	close(exec_fd);
	close(tmp_fd);

	return 0;
}
