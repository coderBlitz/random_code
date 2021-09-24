#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(){
	char filename[21] = "log_temps_F.txt";//Temps pulled from accuweather.com
	printf("Filename is %s\n",filename);

	FILE *fp = fopen(filename,"r");
	char s;
	int lines=0;
	while((s = fgetc(fp)) != EOF) if(s == '\n') lines++;
printf("Counted %d lines\n",lines);

rewind(fp);
	int num,month,day,tmp,unique=0;
	for(int i=0;i<lines;i++){
		int v = fscanf(fp,"%*d:%*d %d/%d/%*d\t%d\n",&month,&day,&num);//'v' is number of args
		if((tmp == day) || v != 3) continue;//Skips if temp for certain day has been found
		printf("Temperature for %d/%d: %d°F\n",month,day,num);
		unique++;
		tmp = day;//Allows skipping multiple temps for same day
	}
	fclose(fp);

	printf("%d unique day temperatures\n",unique);
}
