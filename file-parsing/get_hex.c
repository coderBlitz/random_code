#include<stdio.h>
#include<stdlib.h>

/*This Program is supposed to get the hex code gathered from 'hexdump'
And convert it to alphanumeric characters. Made to get "readable" text from
a Disk image or other sort of odd file. No Idea what this is really useful for
other than maybe getting lost files from a disk image.
*/

int main(){
	char fileName[]="dumpFile.txt";
	FILE *fp = fopen(fileName,"r");

	char ch,bit[4];int col=0,group_pos=0;
	while((ch = fgetc(fp)) != EOF){
		//printf("Ch: %c\tCol: %d\tGroup_pos: %d\n",ch,col,group_pos);
		if(col<8){col++; continue;}
		if(ch == ' '){group_pos=0; continue;}
		if(ch == '\n'){col=0;group_pos=0;continue;}
		switch(group_pos){
			case 0: bit[group_pos]=ch;break;
			case 1: bit[group_pos]=ch;break;
			case 2: bit[group_pos]=ch;break;
			case 3: bit[group_pos]=ch;break;
		}
		if(group_pos == 3){
			for(int i=0;i<4;i++){
				if((bit[i] >= 48) && (bit[i] <= 57)){
					bit[i] -= 48;
				}else if((bit[i] > 96) && (bit[i] < 123)){
					bit[i] -= 87;
				}
			}
			char first = bit[0]*16 + bit[1];
			char second = bit[2]*16 + bit[3];

			first = (first < 0)?0-first:first;
			second = (second < 0)?0-second:second;

		//	printf("First: %d\n",first);
		//	printf("Second: %d\n",second);
			if(second == 10) printf("\n");
			else if(second < 32 || second > 126) printf(".");
			else printf("%c",second);

			if(first == 10) printf("\n");
			else if(first < 32 || first > 126) printf(".");
			else printf("%c",first);
		}
		group_pos++;
	}
	printf("\n");
}
