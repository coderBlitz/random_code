#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void printSVG(char *outFile){
	
}

void getTemps(int *array,char *filename){
	printf("Filename is %s\n",filename);
	FILE *fp = fopen(filename,"r");

	char s;
	int lines=0;
	while((s = fgetc(fp)) != EOF) if(s == '\n') lines++;
printf("Counted %d lines\n",lines);
	array = malloc(lines*sizeof(int));

rewind(fp);
	int num,month,day,tmp,unique=0,hour=0;
	for(int i=0;i<lines;i++){
		int v = fscanf(fp,"%d:%*d %d/%d/%*d\t%d\n",&hour,&month,&day,&num);//'v' is number of args
		if((tmp == day) || v != 4 || !(hour >= 11 && hour <= 23)) continue;//Skips multiple temps in one day, and if time isn't after 11am
//		printf("Temperature for %d/%d: %d°F\n",month,day,num);
		array[i] = num;
		unique++;
		tmp = day;//Allows skipping multiple temps for same day
	}
	fclose(fp);
}

int main(){
	char filename[21] = "log_temps_F.txt";
	
} 
