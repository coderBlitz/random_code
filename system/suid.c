#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

int main() {
    int t;

    printf("before, geteuid() returned %d\n", geteuid());
    printf("before, getuid() returned %d\n", getuid());

    t = setuid(geteuid());
    if (t < 0) {
        perror("Error with setuid() - errno " + errno);
        exit(1);
    }

    printf("after, geteuid() returned %d\n", geteuid());
    printf("after, getuid() returned %d\n", getuid());

    setreuid(geteuid(), geteuid());

    printf("finally, geteuid() returned %d\n", geteuid());
    printf("finally, getuid() returned %d\n", getuid());

    printf("did work fine, look who I am:\n");
    system("/bin/bash -c whoami");
}
