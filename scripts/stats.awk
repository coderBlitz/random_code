#!/bin/awk -f

#	Test file to get basic statistics of myUMBC assignment grade table. First
#	 line won't have titles.
# Column format: grade, num students, ...,

BEGIN{
	FS=",";
	pt_total = 0; # Grade points total
	st_total = 0; # Student total

	st_max = 0;
	st_max_idx = -1;
	st_max_grade = 0;
}

{
	# Update cumulative info
	pt_total += $1 * $2;
	st_total += $2;

	# Save mode info
	if($2 > st_max){
		st_max = $2;
		st_max_idx = NR;
		st_max_grade = $1;
	}

	pts[NR] = $1;
	sts[NR] = $2;
	st_sums[NR] = st_total;
}

END{
	if(st_total == 0){
		exit; # Exit to avoid errors
	}

	# Get mean and mode
	mean = pt_total / st_total;
	mode = st_max_grade;

	# Find median
	for(i = 1;i <= NR;i++){
		if(st_sums[i] > (st_total/2)){
			i--;
			break;
		}
	}
	median = pts[i];

	# Deviations
	variance = 0;
	for(i = 1;i <= NR;i++){
		variance += sts[i] * pts[i] * pts[i];
	}
	variance = variance/st_total - (mean * mean);
	dev = sqrt(variance);

	printf("Total: %f\tStudents: %d\n", pt_total, st_total);
	printf("Mean: %f\tMedian: %f\tMode: %f\n", mean, median, mode);
	printf("Variance: %f\tSigma: %f\n", variance, dev);
}
