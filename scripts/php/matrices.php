<?php
$array = array_fill(0,10,array_fill(0,10,0));

for($i=0;$i<count($array);$i++){
	for($n=0;$n<count($array[$i]);$n++) $array[$i][$n] = $n + $i*count($array);
}

for($i=0;$i<count($array);$i++){
	for($n=0;$n<count($array[$i]);$n++) printf("array[%d][%d]: %d\n",$i,$n,$array[$i][$n]);
}
?>
