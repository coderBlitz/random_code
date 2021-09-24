#include<stdio.h>
#include<stdlib.h>
#include"student.h"

// This is a custom C interpretation of my APCS assignment in java to work with a
// "classroom" object full of "students" objects

int main(){
//	Student s = newStudent("5678","ttftftfttf");// newStudent() creates a student with id and answers
//	printf("Student id: %s\nStudent answers: %s\nStudent score: %hd\n",s.id,s.quizresults,s.score);
//	grade("tttttttttt",&s);// Using answer key, grades Student 's'
//	printf("Graded score: %c (%d)\n",getLetterGrade(&s),s.score);

	FILE *list = fopen("truefalse.txt","r");// Open the file with student stuff

	char c; int lines = -1;// Answer line does not count
	while((c = fgetc(list)) != EOF) if(c == '\n') lines++;
	rewind(list);// Reset to get student data

	char KEY[11];// Hold the answer key
	for(int i=0;i<10;i++) KEY[i] = fgetc(list);
	fgetc(list);// skip newline

	printf("ANSWER KEY: %s\n",KEY);

	char id[lines][5];// Holds the student IDs
	char answers[lines][11];// Holds the student's answers
	for(int n=0;n<lines;n++){
		fscanf(list,"%s %s",id[n],answers[n]);
	}
	fclose(list);// Close since the file is no longer needed

	Student classroom[lines];// Hold each students data
	for(int i=0;i<lines;i++){
		classroom[i] = newStudent(id[i],answers[i]);// Add each student to an array
		grade(KEY,&classroom[i]);
	}

	for(int i=0;i<lines-1;i++){// Orders scores high to low in 'classroom'
		if(classroom[i].score < classroom[i+1].score){
			Student tmp = classroom[i];// Get student
			classroom[i] = classroom[i+1];// Switch first
			classroom[i+1] = tmp;// Switch second
			i=-1;
		}else{
			continue;
		}
	}
// Print scores of every student
	for(int i=0;i<lines;i++)printf("%s: %c (%d)\n",
						classroom[i].id,
						getLetterGrade(&classroom[i]),
						classroom[i].score);
}

