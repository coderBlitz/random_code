#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define SIZE 20

 short *open;// This will hold the indices of open parenthesis for all methods

void removeWhite(char *eq){
	int len = strlen(eq);
	printf("EQ: \"%s\"\tEQ LENGTH: %d\n",eq,len);
	for(int i=0;i<len;i++){
		if(eq[i] == 32 || eq[i] == 10){
		printf("Removing at %d\n",i);
			for(int j=i;j<len-1;j++){
				eq[j] = eq[j+1];
			}
		}
	}
	printf("New string \"%s\"",eq);
}

void calculate(char *eq,double *total){// Calculate normal equations 'eq'= equation; 'total'= just that
	int count=0;// Gets paranthesis group # correct on any amount of groupings
	for(int i=(SIZE/2)-1;i>=0;i--){// Go through parenthesis first
		if(open[i] == -1){ count++; continue; }// Skip no parenthesis

		int start=open[i],end;//				  These two set the parenthesis
		char skip = (SIZE/2)-count - i,tmp = 1;
printf("SKIPPPPPING: %d\n",skip);
		for(int n=open[i];n<SIZE;n++){
			if(eq[n] == ')'){
				if(tmp == skip){
					end=n;
					break;
				}else{
					tmp++;
				}
			}// bounds for solving
		}

			printf("Parenthesis group %d: %.*s\n",((SIZE/2)-count)-i, (end-start-1), &eq[start+1]);

			int range = end-start; printf("range: %d\n",range);
			char group[range];// Temporarily separate everything in this set of parenthesis
			sprintf(group,"%.*s",(range-1), &eq[start+1]);

			int addition=0,subtraction=0,multiply=0,divide=0;
			for(int n=0;n<range;n++){// Check operations present
				if(group[n] == '*') multiply=1;
				if(group[n] == '/') divide=1;
				if(group[n] == '+') addition=1;
				if(group[n] == '-') subtraction=1;
			}

		// *************** get numbers *******************
			int ops=0;
			for(int n=0;n<range;n++){// Counts number of operations in group
				if(group[n] > 41 && group[n] < 48) ops++;
			}
			double nums[ops+1]; int lastOp=0,count=0;// Always will be 1+numberOfOps
			for(int n=0;n<range;n++){
				sscanf(&group[lastOp],"%lf",&nums[count]);
				if(group[n] > 41 && group[n] < 48){
					lastOp=n+1; count++;
					printf("Number %d found: %lf\n",count,nums[count-1]);
				}
			}
			printf("Number %d found: %lf\n",count+1,nums[count]);

// **************************** Operate on Numbers *****************************/
// Need to add array system for adding and subtracting numbers. Also need to add
// system to replace numbers in equation string, so following operations work properly
			ops=0;
			for(int n=0;n<range;n++){
				if(group[n] > 41 && group[n] < 48){
					ops++;
					if(group[n] == '*' && ops == 1){// Multiplication
						(*total) += nums[1]*nums[0]; continue;
					}else if(group[n] == '*'){ (*total) *= nums[ops]; continue; }

					if(group[n] == '/' && ops == 1){// Division
						(*total) += nums[0]/nums[1]; continue;
					}else if(group[n] == '/'){ (*total) /= nums[ops]; continue; }

					if(group[n] == '+' && ops == 1){// Addition
						(*total) += nums[0]+nums[1];
					}else if (group[n] == '+'){ (*total) += nums[ops]; continue; }
				}
			}

// *****************************************************************************/
		for(int n=open[i];n<end;n++){// Go from start of parenthesis to end, remove numbers
			eq[n] = 32;// Make it white space so method above gives correct length
		}
		removeWhite(eq);
		printf("Partial Total = %lf\n",*total);
	}
	printf("Equation is '%s'\n",eq);
}


int main(){// ******************* MAIN METHOD *************************/
	int pairs=0;
	char *equation = (char *)malloc(sizeof(char) * (SIZE+1));
	open = malloc((SIZE/2) * sizeof(short));// Allocate memory for '*open'
	for(int i=0;i<(SIZE/2);i++) open[i] = -1;

	printf("Equation Solver 0.1\nUse 'X' as variable if needed\nEnter equation: ");
	fgets(equation, (sizeof(equation)-1)/2, stdin);// Either change 20 with SIZE, or remove completely
printf("Entered Equation: \"%s\"\n",equation);

printf("checking pairs...\n");// ***************** Checks syntax *************************
	removeWhite(equation);
	int hasX=0,nextOpen=0;// 'nextOpen' is counter for 'open[]'
	for(int i=0;i<strlen(equation);i++){// Check parenthesis
		if(pairs == 0 && equation[i] == ')'){
			printf("Mismatching parenthesis at: %d\n",i+1);
			exit(1);
		}else if(equation[i] == '('){ pairs++; open[nextOpen] = i; nextOpen++; }
		else if(equation[i] == ')') pairs--;
		else if(equation[i] == 'X') hasX=1;// Decide between regular and monomial/polynomial
		else if(equation[i] == 'x'){ hasX=1; equation[i]='X'; }//Capitalize all X's
	}
	printf("Pairs: %d\n",pairs);
	if(pairs != 0){
		printf("Missing %d parenthesis\n",pairs);
		exit(1);
	}

	double total=0;
	if(hasX){ printf("It HasX!\n"); }// Add method to solve monomials
	else{// Uses calculate(equation, total) to solve regular equations
		printf("It does not HasX!\n");
		calculate(equation,&total);
		printf("Solution: %lf\n",total);
	}

//*************************** Free allocated memory **********************/
	free(open);
	free(equation);
}

