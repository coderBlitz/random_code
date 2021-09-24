#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/sysinfo.h>

int main(){
	struct sysinfo s;

	sysinfo(&s);

	printf("Uptime: %ld\nTotal mem: %lu\nFree mem: %lu\n%lu\n%lu\n",s.uptime,s.totalram,s.freeram,s.bufferram,s.sharedram);
	unsigned long total = s.freeram;
	printf("Total available: %luMB\n",total/(1024*1024));

	long *a = malloc(total);
	if(a == NULL) printf("Too little mems!\n");
	else{
		printf("Filling %ld..\n",total/sizeof(*a));
		for(long i=0;i < total/sizeof(*a);i++) a[i] = i;
		sleep(5);
	}
	free(a);
}
