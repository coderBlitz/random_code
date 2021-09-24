#ifndef STUDENT_H_
#include<string.h>
#define STUDENT_H_ // Header file for "student.c"

typedef struct student_struct{
	char id[5];
	char quizresults[11];// 10 answers, plus null terminator '\0'
	short score;
} Student ;

Student newStudent(char *const a,char *const answers);

char getLetterGrade(Student *s);

void grade(char answers[],Student *s);
#endif
