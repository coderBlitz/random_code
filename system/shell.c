/***
	Author		: Chris Skane
	Created		: 20190227
	Modified	: 20200221
	Description	: A basic shell
***/

#include<errno.h>
#include<limits.h>
#include<pwd.h>
#include<signal.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/wait.h>
#include<unistd.h>
#include"utility.h"

#define _BUFFER_INIT_SIZE 0x20
#define PROMPT_LEN 64

#ifndef USE_CUSTOM_PROMPT
	#define USE_CUSTOM_PROMPT 1
#endif

extern char **environ;

static const char *DEFAULT_PROMPT = "$ "; // Default prompt, also used with custom
static char PROMPT[PROMPT_LEN]; // Buffer to store prompt
static pid_t child_pid = 0; // To get prompt to print when process interrupted
static int8_t BAD_RUN = 0; // Flag for parent
static int8_t CHILD_EXIT = 0; // Flag for child


/**	SIGINT handler function
**/
void interrupt(int err){
	printf("\n");
	if(!child_pid) printf("%s", PROMPT);
	fflush(stdout);
	fflush(stdin);
}

/**	Simply catches the childs signal, to notify parent of failed exec
**/
void badRunHandler(int err){
	//printf("Child went to grave.\n");
	BAD_RUN = 1;
}

/**	Given a valid exec* list of arguments, check for built-in command
	Return negative if no built-in exists, so normal exec* should be called
	Return positive (including zero) value indicating built-in rv
**/
int8_t shell_builtins(char **args){
	if(args == NULL || args[0] == NULL){
		fprintf(stderr, "Invalid argument list given to shell_builtins\n");
		return -1;
	}

	// Checks for "exit" statement
	if(!strcmp(args[0], "exit")){
		//printf("builtin exit called\n");
		long rv = 0;

		// If there is a first argument, attempt to parse a number from it
		//  and use it for the return value.
		if(args[1] != NULL){
			char *loc = NULL;
			long temp = strtol(args[1], &loc, 10);
			if(loc != args[1]){
				rv = temp;
			}
		}

		CHILD_EXIT = 1; // This allows parent to gracefully and quietly exit
		return rv;
	}else if(!strcmp(args[0], "cd")){
		if(args[1] != NULL){
			int rv = chdir(args[1]);
			if(rv){
				perror("Could not change directory");
				return 1;
			}

			return 0;
		}else{
			// Default to the home directory if no argument given

			struct passwd *pwd = getpwuid(getuid()); // Fetch PASSWD DB entry

			int rv = chdir(pwd->pw_dir); // Use PWD entry
			//int rv = chdir(getenv("HOME")); // Use environment for home
			if(rv){
				perror("Could not change to home directory");
				return 1;
			}

			return 0;
		}
	}else if(!strcmp(args[0], "pwd")){ // Not necessary, but faster this way
		printf("%s\n", getcwd(NULL, 0)); // Print directory

		return 0;
	}

	return -1;
}

/** Given a single string with command and arguments, it
	will parse and execute the first token in the string
	passing the remaining tokens as arguments
**/
int8_t shell_parse_command(char *const restrict command){
	if(command == NULL){
		fprintf(stderr, "No command given\n");
		return 1;
	}

	int ret = 0;


	// Get arguments, escaping as they are split/parsed
	char **args = split_args(command);
	if(args == NULL || args[0] == NULL){
		fprintf(stderr, "Received no valid command/arguments.\n");
		free(args);
		//free(command);

		BAD_RUN = 1;
		return 1; // Not an error, but not a success, so 1
	}

	// Check if command is built-in before attempting to execute binary
	ret = shell_builtins(args);
	//printf("builtin ret: %d\n", ret);
	if(CHILD_EXIT || ret >= 0){
		free(args);
		return ret;
	}

	// If the command isn't an absolute or relative path, search PATH
	char *temp = NULL;
	if(args[0][0] != '/' && args[0][0] != '.'){
		temp = shell_search_path(args[0]);

		if(temp != NULL){
			args[0] = temp;
		}else{
			// Return here, to not run files in current directory
			fprintf(stderr, "Command not found\n");

			free(args);
			//free(command);

			BAD_RUN = 1;
			return 1; // Not an error, but not a success, so 1
		}
	}


	uid_t pid = 1;
	BAD_RUN = 0; // Reset flag
	pid = fork();

	if(pid == -1){
		perror("Could not run create child process");
		BAD_RUN = 1;
		return 1;
	}else if(!pid){
		pid_t ppid = getppid();

		signal(SIGINT, SIG_DFL); // Un-handle interrupt, so process stops as expected
		//printf("Executing '%s'\n", args[0]);
		ret = execve(args[0], args, environ);
		if(ret == -1){
			perror("Failed to execute");
			//fprintf(stderr, "Failed to execute '%s': %s\n", args[0], strerror(errno));

			free(args);
			//free(command);
			free(temp);

			CHILD_EXIT = 1;
			kill(ppid, SIGUSR1);
			return -8; // Return after setting error flag, so memory can be freed
		}
	}else{
		free(args); // These are not used after fork, so free immediately
		free(temp);


		int status = 0;

		child_pid = pid;
		ret = waitpid(pid, &status, WUNTRACED);
		child_pid = 0;
		//printf("Ret: %d\n", ret);

		if(ret == -1){
			perror("Wait error");
		}

		if(WIFEXITED(status)){
			//printf("Process returned normally: %d\n", WEXITSTATUS(status));
			return WEXITSTATUS(status);
		}
	}

	return 0;
}

/** Given a filename, it will attempt to open the file
	and execute each line until the end of the file, or
	the failure of an execution
**/
int8_t shell_file(const char *restrict fname){
	if(fname == NULL){
		fprintf(stderr, "No filename given\n");
		return 1;
	}

	FILE *fp = fopen(fname, "r");
	if(fp == NULL){
		perror("Could not open file");
		return 1;
	}

	// Allocate line buffer
	uint64_t buffer_len = _BUFFER_INIT_SIZE;
	char *buffer = malloc(buffer_len * sizeof(*buffer));
	if(buffer == NULL){
		fprintf(stderr, "Could not allocate buffer");
		fclose(fp);
		return 1;
	}

	// Loop to get each line, then pass to command parser
	uint64_t line = 0;
	char c = 0;
	int8_t ret = 0;
	uint64_t idx = 0;
	while((c = fgetc(fp)) != EOF){
		// Reallocate buffer with twice the size, if necessary
		if(idx == buffer_len){
			buffer_len <<= 1;
			char *temp = NULL;
			temp = realloc(buffer, buffer_len*sizeof(*buffer));
			if(temp == NULL){
				// Realloc failed
				fprintf(stderr, "Could not allocate more space for buffer\n");
				free(buffer);
				fclose(fp);
				return 1;
			}
			buffer = temp;
		}

		if(c == '\n' || c == '\0'){
			buffer[idx] = '\0';
			line++;

			if(idx == 0) continue; // Skip empty lines
			ret = shell_parse_command(buffer);

			// CHILD_EXIT is the "graceful" exit condition
			// BAD_RUN is the exit-on-failure condition
			if(CHILD_EXIT){
				free(buffer);
				fclose(fp);
				exit(ret);
			}else if(BAD_RUN){
				fprintf(stderr, "Error on line %d\n", line);

				free(buffer);
				fclose(fp);
				return 1;
			}else if(c == '\0'){ // This is probably unnecessary, given outer condition
				free(buffer);
				fclose(fp);
				return ret;
			}

			idx = 0;
		}else{
			buffer[idx] = c;
			++idx;
		}
	}

	// Cleanup
	fclose(fp);
	fp = NULL;

	free(buffer);
	buffer = NULL;

	return 0;
}

/** The default behavior of the shell is to open for
	direct user interaction, parsing each line similar
	to reading from a file
**/
int8_t shell_interactive(void){
	// Line buffer
	uint64_t buffer_len = _BUFFER_INIT_SIZE;
	char *buffer = malloc(buffer_len * sizeof(*buffer));
	if(buffer == NULL){
		fprintf(stderr, "Could not allocate buffer");
		return 1;
	}

	// Setup the prompt (make construction helper at some point)
	// Currently make prompt with username in front
	if(USE_CUSTOM_PROMPT){
		struct passwd *ent = getpwuid(getuid());
		if(ent == NULL){
			fprintf(stderr, "Could not fetch username.\n");
			strcpy(PROMPT, DEFAULT_PROMPT);
		}else{
			// 3 extra characters. Not important, everything is manually done
			int prompt_remain_len = PROMPT_LEN - strlen(DEFAULT_PROMPT) - 3;

			strcpy(PROMPT, "[");
			strncat(PROMPT, ent->pw_name, prompt_remain_len);
			strcat(PROMPT, "] ");
			strcat(PROMPT, DEFAULT_PROMPT);
			//printf("Using prompt: '%s'\n", PROMPT);
		}
	}else{
		strcpy(PROMPT, DEFAULT_PROMPT);
	}

	char c = 0;
	int8_t ret = 0;
	uint64_t idx = 0;

	printf("%s", PROMPT); // Could possibly re-order loop print lines to remove this
	do{
		// Reallocate buffer with twice the size, if at max
		if(idx == buffer_len){
			buffer_len <<= 1;
			char *temp = NULL;
			temp = realloc(buffer, buffer_len*sizeof(*buffer));
			if(temp == NULL){
				// Realloc failed
				fprintf(stderr, "Could not allocate more space for buffer\n");
				return 1;
			}
			buffer = temp;
		}

		c = getc(stdin);
		if(c == '\n'){
			buffer[idx] = '\0';

			if(buffer[0] != '\0') ret = shell_parse_command(buffer);

			if(CHILD_EXIT){
				free(buffer);
				exit(ret);
			}

			printf("%s", PROMPT);
			idx = 0;
		}else if(c == EOF){ // Handles the Ctrl-D sequence, which closes stdin
			printf("\n");
			break;
		}else{
			buffer[idx] = c;
			++idx;
		}
	}while(1);

	// Cleanup
	free(buffer);
	buffer = NULL;
	return 0;
}

/** Main simply handles checking all arguments, and
	determining whether to parse from a file or to
	go to an interactive user session
**/
int main(int argc, char *argv[]){
	int ret = 0;

	// Custom signal used to tell parent if exec failed in child process
	signal(SIGUSR1, badRunHandler);

	// Process arguments
	if(argc > 2){
		fprintf(stderr, "Usage: simple_shell [script]\n");
		return 1;
	}else if(argc == 2){
		for(uint32_t i = 1;i < argc;++i){
			uint16_t len = strlen(argv[i]);
			// Parse file argument (run single file for now)
			if(len > NAME_MAX){
				fprintf(stderr, "File name too long: %u\n", len);
				return 1;
			}else{
				return shell_file(argv[i]);
			}
		}
	}else{
		// Interactive if no file given
		signal(SIGINT, interrupt); // We want to handle ctrl-c when interactive
		printf("Starting interactive session\n");
		shell_interactive();
	}

	return 0;
}
