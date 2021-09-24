#include<stdio.h>
#include<stdlib.h>
#include<ncurses.h>//Compile with option -l curses or -lcurses

int main(){
	initscr();//Gives a blank screen
	noecho();//Doesn't print character when pressed
	cbreak();//Doesn't wait for Enter to be pressed
	timeout(100);
	while(1){
		char c = (char)getch();//getch() is part of conio.h
		if(c == -1) continue;
		printf("Entered: %c\n\r",c);
	}
	endwin();
	nocbreak();//Returns to normal, Wait-for-Enter buffer
}
