#include"student.h"


Student newStudent(char *const a,char *const answers){// Creates Student struct with id and answers
	Student s;

	strcpy(s.id,a);// Copy student id
	strcpy(s.quizresults,answers);// Copy answers (not plagarism)

	return s;
}

char getLetterGrade(Student *s){// Returns A,B,C,D, or F
	switch(s->score){// Get score from Student 's'
			case(100): return 'A';
			case(90): return 'B';
			case(80): return 'C';
			case(70): return 'C';
			case(60): return 'D';
			case(50): return 'D';
		}
		return 'F';
}

void grade(char answers[],Student *s){// Updates Student score
	short count = 0;

	for(int i=0;i<strlen(answers);i++){
		if(answers[i] == s->quizresults[i]) count += 10;
	}
	s->score = count;
}
