#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* reorder an array to put elements <= pivot
 * before elements > pivot.
 * Returns number of elements <= pivot */
static int
splitByPivot(int n, int *a, int pivot)
{
    int lo;
    int hi;
    int temp;  /* for swapping */

    assert(n >= 0);

    /* Dutch Flag algorithm */
    /* swap everything <= pivot to bottom of array */
    /* invariant is i < lo implies a[i] <= pivot */
    /* and i > hi implies a[i] > pivot */
    lo = 0;
    hi = n-1;

    while(lo <= hi) {
        if(a[lo] <= pivot) {
            lo++;
        } else {
            temp = a[hi];
            a[hi--] = a[lo];
            a[lo] = temp;
        }
    }

    return lo;
}

/* find the k-th smallest element of an n-element array */
/* may reorder elements of the original array */
int
quickselectDestructive(int k, int n, int *a)
{
    int pivot;
    int lo;

    assert(0 <= k);
    assert(k < n);

    if(n == 1) { 
        return a[0];
    }
    
    /* else */
    pivot = a[rand() % n];   /* we will tolerate non-uniformity */

    lo = splitByPivot(n, a, pivot);

    /* lo is now number of values <= pivot */
    if(k < lo) {
        return quickselectDestructive(k, lo, a);
    } else {
        return quickselectDestructive(k - lo, n - lo, a + lo);
    }
}

/* sort an array in place */
void
quickSort(int n, int *a)
{
    int pivot;
    int lo;

    if(n <= 1) { 
        return;
    }
    
    /* else */
    pivot = a[rand() % n];   /* we will tolerate non-uniformity */

    lo = splitByPivot(n, a, pivot);

    quickSort(lo, a);
    quickSort(n - lo, a + lo);
}


/* shuffle an array */
void
shuffle(int n, int *a)
{
    int i;
    int r;
    int temp;

    for(i = n - 1; i > 0; i--) {
        r = rand() % i;
        temp = a[r];
        a[r] = a[i];
        a[i] = temp;
    }
}

#define N (1024)

int
main(int argc, char **argv)
{
    int a[N];
    int i;

    srand(0);  /* use fixed value for debugging */

    for(i = 0; i < N; i++) {
        a[i] = i;
    }

    shuffle(N, a);

    for(i = 0; i < N; i++) {
        assert(quickselectDestructive(i, N, a) == i);
    }

    shuffle(N, a);

    quickSort(N, a);

    for(i = 0; i < N; i++) {
        assert(a[i] == i);
    }

    return 0;
}
