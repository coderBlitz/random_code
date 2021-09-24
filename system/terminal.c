/*	terminal.c -- Mess around with terminal settings and termios structure.
	Author: Chris Skane
	Notes:
		For reference see "Beginning Linux programming" 4th edition, chapter 5
	Control key codes are
		Ctrl-` => 0
		Ctrl-a => 1
		Ctrl-b => 2
		...
		Ctrl-l => 12
		Ctrl-m => 10
		Ctrl-n => 14
		...
		Ctrl-z => 26
*/

#include<fcntl.h>
#include<signal.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<termios.h>
#include<unistd.h>

int tty;
struct termios term_orig;
/*	Struct definition:
	struct termios {
		tcflag_t c_iflag;	// Input flag
		tcflag_t c_oflag;	// Output flag
		tcflag_t c_cflag;	// Control flag
		tcflag_t c_lflag;	// Local flag
		cc_t c_cc[NCCS];	// Special control characters
	};
*/

void interrupt(int num){
	printf("\nAwwwww, you're no fun.\n");

	tcsetattr(tty, TCSANOW, &term_orig);
	close(tty);

	exit(1);
}

int main(int argc, char *argv[]){
	signal(SIGINT, interrupt);
	struct termios term = {0};

	// Open controlling terminal
	tty = open("/dev/tty",  O_RDWR);
	if(tty < 0){
		fprintf(stderr, "Something failed.\n");
		exit(1);
	}

	// Get terminal settings
	tcgetattr(tty, &term_orig);
	term.c_iflag = term_orig.c_iflag;
	term.c_oflag = term_orig.c_oflag;
	term.c_cflag = term_orig.c_cflag;
	term.c_lflag = term_orig.c_lflag;

	// Set desired properties
	term.c_lflag &= ~(ECHO | ICANON);
	term.c_cc[VINTR] = 0x3; // Ctrl-C
	term.c_cc[VMIN] = 1; // Wait for 1 character
	//term.c_cc[VEOF] = 4; // Ctrl-D
	tcsetattr(tty, TCSANOW, &term);

	printf("Type some stuff: ");
	char c[9];
	int res = 0;
	do{
		res = read(0, c, 8);
		c[res] = 0; // Add terminator for sequences

		// Sequences always get multiple in a single read. Unlikely to be mashed keys
		if(c[0] == 27){
			if(res == 3){
				// Arrow keys, f1-f4, home/end
				if(c[1] == '['){
					// Arrows, home/end
					switch(c[2]){
						case 'A':
						// Up
						printf("Up\n");
						break;
						case 'B':
						// Down
						printf("Down\n");
						break;
						case 'C':
						// Right
						printf("Right\n");
						break;
						case 'D':
						// Left
						printf("Left\n");
						break;
						case 'F':
						// End
						printf("End\n");
						break;
						case 'H':
						// Home
						printf("Home\n");
						break;
					}
				}else if(c[1] == 'O'){
					// Fn keys
				}
			}else if(res == 4){
				// Insert/delete, PgUp/PgDn
			}else if(res == 5){
				// f5-f10, f12 (f11 fullscreens automatically)
			}else if(res == 6){
				// Multiple Ctrl and/or alt modifiers
			}else if(res == 7){
				// Ctrl-f5, Ctrl-f6
			}
			printf("Control sequence %d: ", res);
			for(int i = 0;i < res;++i) printf("%c (%d)\n", c[i], c[i]);
		}else{
			printf("You entered %d: '%c' (%d)\n", res, c[0], c[0]);
		}
		printf("Type some more: ");
		fflush(stdout);
	}while(1);

	// Restore original settings
	tcsetattr(tty, TCSAFLUSH, &term_orig);
	close(tty);

	return 0;
}
