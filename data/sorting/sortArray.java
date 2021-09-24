import java.util.Random;

public class sortArray{
	// Sorts numbers in 'array' of length 'SIZE' is descending order
	static void sortDesc(int array[],final int SIZE){
		for(int i=0;i<SIZE;i++){
			if(array[i] >= array[i+1]) continue;
			else{
				int tmp = array[i];
				array[i] = array[i+1];
				array[i+1] = tmp;
				i=-1;
			}
		}
	}

	// Sorts numbers in 'array' of length 'SIZE' is ascending order
	static void sortAsc(int array[],final int SIZE){
		for(int i=0;i<SIZE-1;i++){
			if(array[i] <= array[i+1]) continue;
			else{
				int tmp = array[i];
				array[i] = array[i+1];
				array[i+1] = tmp;
				i=-1;
			}
		}
	}

	public static void main(String args[]){
		Random rand = new Random(System.currentTimeMillis());
		int SIZE = 15;
		int arr[] = new int[SIZE];
		for(int i=0;i<SIZE;i++) arr[i] = rand.nextInt(51);

		System.out.printf("Array before:");
		for(int i=0;i<SIZE;i++) System.out.printf(" %d",arr[i]);
		System.out.printf("\n");

		sortAsc(arr,SIZE);// sortAsc(array,arraySize);

		System.out.printf("Array after:");
		for(int i=0;i<SIZE;i++) System.out.printf(" %d",arr[i]);
		System.out.printf("\n");
	}
}
