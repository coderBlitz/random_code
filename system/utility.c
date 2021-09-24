#include"utility.h"


char *shell_search_path(char *const file){
	char *temp = getenv("PATH");
	if(temp == NULL){
		return NULL;
	}

	// Allocate separate path buffer, to not mess with environment
	char *PATH = calloc(strlen(temp)+1, sizeof(*PATH));
	if(PATH == NULL){
		fprintf(stderr, "Could not allocate path buffer");
		return NULL;
	}
	//for(int i = 0;i < strlen(temp)+1;i++) PATH[i] = 0; // Fixes valgrind warnings
	strncpy(PATH, temp, strlen(temp));

	char *dir = NULL;
	char *full = NULL;

	// Buffer to store path + filename, to be tested for existence
	uint64_t buffer_len = NAME_MAX+1;
	char *buffer = malloc(buffer_len * sizeof(*buffer));
	if(buffer == NULL){
		fprintf(stderr, "Could not allocate path search buffer");
		free(PATH);
		return NULL;
	}

	// Split up variable into each entry
	dir = strtok(PATH, ":");
	FILE *fp = NULL;
	uint32_t path_len = strlen(dir);

	// Re-allocation loop. Size up buffer to fit path + file
	while(path_len + (NAME_MAX+1) >= buffer_len){
		buffer_len <<= 1;
		temp = realloc(buffer, buffer_len*sizeof(*buffer));
		if(temp == NULL){
			// Realloc failed
			fprintf(stderr, "Could not allocate more space for path search buffer\n");
			free(PATH);
			free(buffer);
			return NULL;
		}
		buffer = temp;
	}

	// Iterate for each path entry
	while(dir != NULL){
		for(int i = 0;i < buffer_len;i++) buffer[i] = 0; // Fixes valgrind warnings

		// Construct full path
		strncpy(buffer, dir, path_len);
		strcat(buffer, "/");
		++path_len;
		strncat(buffer, file, 255);

		// Test file existence
		//printf("File: '%s'\n", buffer);
		if(access(buffer, F_OK) != -1){
			//printf("File found at '%s'\n", buffer);
			free(PATH);
			return buffer;
		}

		dir = strtok(NULL, ":");
	}

	free(buffer);
	free(PATH);
	return NULL;
}

char **split_args(char *command){
	int8_t quoted = 0;

	// Whitespace flag, with prev value. Prev value simplifies arg conditional. 1 for first arg
	/**** Whitespace MUST be set to 1. So conditional adds command to args ****/
	int8_t whitespace = 1, prevWSval = 0;

	char *cur = command; // Current character that will be placed in argument
	char *iter = command; // Current character that is being escaped, or otherwise
	uint32_t arg = 0; // Current argument number/position

	size_t arg_buffer_size = 16;
	char **args = malloc(arg_buffer_size*sizeof(*args));
	args[0] = command;

	/*
		If un-quoted, non-whitespace, and follows a whitespace, next arg starts
		If quoted, check for "\\", otherwise append to current arg
		If '\', check next char for digit, 'x', or all others. Then escape
		If quotation, toggle quoted flag
		Argument ends at first unquoted, unescaped whitespace
	*/
	while(*iter){
		prevWSval = whitespace;
		if(*iter == ' ' || *iter == '\t') whitespace = 1;
		else whitespace = 0;

		if(!quoted && prevWSval && !whitespace){
			args[arg++] = cur;
			//printf("Arg %d starts at %d.\tcur: %d\n", arg, iter-command, cur-command);
		}else if(!quoted && !prevWSval && whitespace){
			//printf("Arg %d ends at %d.\tcur: %d\n", arg, iter-command, cur-command);
			*cur++ = 0; // End of argument
			iter++;
			continue;
		}

		// If quotation, mark. If escape, handle. Otherwise add character as-is
		if(*iter == '"' || *iter == '\''){
			quoted = (quoted) ? 0 : *iter; // toggle
		}else if(*iter == '\\'){
			// Decide if next is hex, octal, "\\", or other
			// If quoted, only check for "\\" escape
			char c = *(++iter);
			//printf("Checking escape for '%c'\n", c);
			if(quoted){
				// Only escape the quotation delimeter, everything else stays
				if(c != quoted) *cur++ = '\\';
			}else if(c >= '0' && c <= '7'){
				// Octal
				c = get_escape_octal(iter);
				if(c > 07) iter++;
				if(c > 077) iter++;
			}else if(c == 'x' || c == 'X'){
				// Hex
				//printf("Hex\n");
				c = get_escape_hex(++iter);
				if(c > 0xF) iter++;
			}else{
				// Normal
				c = get_escape_char(&c);
			}

			if(c > 0) *cur++ = c;
			else{
				fprintf(stderr, "Invalid escape sequence\n");
				//iter++;
			}
		}else{
			*cur++ = *iter;
		}

		// Expand argument array when needed
		if(arg == (arg_buffer_size-1)){
			arg_buffer_size <<= 1;
			char **temp = realloc(args, arg_buffer_size*sizeof(*args));
			if(temp == NULL){
				// Realloc failed
				fprintf(stderr, "Could not allocate more space for args\n");

				free(args);
				return NULL;
			}
			args = temp;
		}

		iter++;
	}
	*cur = 0;
	//printf("to run: '%s' with %d args\n", command, arg);
	args[arg] = NULL; // Denote end of argument list

	//for(int i = 0;i < arg;i++) printf("arg[%d]: '%s'\n", i, args[i]);

	// If quoted after loop, return NULL for mis-matched quotations
	if(quoted){
		fprintf(stderr, "Mis-matched quotations in command.\n");
		free(args);
		return NULL;
	}

	return args;
}

/** Helper function
**/
char get_escape_char(char *const c){
	switch(*c) {
		case '\0':
			return -1;
		case 'n':
			return '\n';
		case 'a':
			return '\a';
		case 'b':
			return '\b';
		case 'r':
			return '\r';
		case '\\':
			return '\\';
		case 'f':
			return '\f';
		case 'v':
			return '\v';
		case '\'':
			return '\'';
		case '"':
			return '"';
		case '?':
			return '?';
		case '*':
			return '*';
		case '$':
			return '$';
		case 't':
			return '\t';
		case ' ':
			return ' ';
		case '!':
			return '!';

		default:
			return *c;
	}

	return 0;
}


char get_escape_octal(char *str){
	int8_t count = 0;
	for(;count < 3;count++){
		if(str[count] < '0' || str[count] > '7') break;
	}
	count--;
	//printf("%d octal digits\n", count+1);

	int res = 0;
	for(int i = 0;count >= 0;count--, i++){
		res |= (str[i] - '0') << (3*count);
	}

	//printf("Res: %d (%c)\n", res, res);
	return (char)res;
}


char get_escape_hex(char *const str){
	if(*str < '0')
		return -1;
	else if(*str > '9' && *str < 'a')
		return -1;
	else if(*str > 'f' && *str < 'A')
		return -1;
	else if(*str > 'F')
		return -1;

	int8_t count = 0;

	char cur = str[1];
	if(cur >= '0' && cur <= '9')
		count++;
	else if(cur >= 'a' && cur <= 'f')
		count++;
	else if(cur >= 'A' && cur <= 'F')
		count++;

	//printf("%d hex digits\n", count+1);

	int res = 0;
	for(int i = 0;count >= 0;i++,count--){
		cur = str[i];
		if(cur >= '0' && cur <= '9')
			res |= (cur - '0') << (4*count);
		else if(cur >= 'a' && cur <= 'f')
			res |= (cur - 'a' + 10) << (4*count);
		else if(cur >= 'A' && cur <= 'F')
			res |= (cur - 'A' + 10) << (4*count);
	}

	return (char) res;
}
