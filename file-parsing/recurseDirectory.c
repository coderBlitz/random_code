#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>
#include<string.h>

int main(){
	const char dirName[] = "/home/chris/Documents";// Directory to be opened
	char cwd[257]; strcpy(cwd,dirName);
	DIR *direct = opendir(dirName);// Open directory
	if(direct == NULL){// Check if it exists
		printf("Could not open directory \"%s\"\n",dirName);
		exit(1);
	}

	struct dirent *dirptr = NULL;// Pointer to store filenames in Directory

//	printf("Files in \"%s\":\n",dirName);
	int n=0,originalPos=0;;
	while(1){// **************************** CURRENTLY LOOPS INFINITELY **************************
		printf("CWD: %s\n",cwd);
		if(n != 0) direct = opendir(cwd);
		for(int i=0;i<originalPos;i++) readdir(direct);// Skips files already read

		while((dirptr = readdir(direct)) != NULL){// While there are still files,
//			printf("%d: %s\n",n,dirptr->d_name);// Print the file name 'd_name' from dirptr struct
			if(strcmp((dirptr->d_name),".") == 0) continue;
			else if(strcmp((dirptr->d_name),"..") == 0) continue;
			char tmp[257];
// Maybe combine following 3 strcat's in one sprintf()
			strcpy(tmp,cwd);// Get Current directory
			strcat(tmp,"/");// Append '/'
			strcat(tmp,dirptr->d_name);// Add filename
		printf("Testing file \"%s\" for directory\n",tmp);
			DIR *tmpDir = opendir(tmp);

			if(tmpDir != NULL && (dirptr = readdir(tmpDir)) != NULL){
				strcpy(cwd,tmp);
				printf("Changing directory to \"%s\"\n",cwd);
				n++;
				break;
			}
			closedir(tmpDir);
			n++;
		}

		if(strcmp(cwd,dirName) == 0) break;// If in original directory, and above loop exits, we're done here

		for(int i=strlen(cwd);i >= strlen(dirName);i--){// Go back to previous directory
			if(cwd[i] == '/'){// Find previous slash denoting the previous directory
				strncpy(cwd,cwd,i);// Copy everything until that slash
			printf("Going back to \"%s\"\n",cwd);
				originalPos++;
				break;// Exit loop to go back to the top
			}
		}

		closedir(direct);// Close the directory
	}


	printf("There are %d files in directory tree \"%s\"\n",n,dirName);
}
