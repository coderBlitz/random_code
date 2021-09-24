<?php
function factorial($num){
	$total = 1;
	for ($i= 1;$i<$num+1;$i++) $total *= $i;
	return $total;
}

$N = readline("Enter number to find factorial: ");
$total = factorial($N);
printf("The factorial of %d is %d\n",$N,$total);
?>
