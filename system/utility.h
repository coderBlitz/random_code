#ifndef UTILITY_H_
#define UTILITY_H_

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<limits.h>
#include<unistd.h>


/**	Given a command name, searches the PATH environment variable for the first
	existing file.
**/
char *shell_search_path(char *const file);

/** Argument splitter. Possibly escapes stuff too
	Doesn't currently return number of args. Could make parameter and/or flip return
**/
char **split_args(char *);

char get_escape_char(char *const);

/**	Get pointer to start of octal digits (1,2 or 3), return converted char
	Can easily be generalized
**/
char get_escape_octal(char *const);

/**	Gets pointer to start of hex characters, returns converted char
	Only works for 1-2 characters
**/
char get_escape_hex(char *const);

#endif
