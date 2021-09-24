#include<gmp.h>
#include<stdint.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
	int num = 52;

	mpz_t val;
	mpz_init(val);

	mpz_fac_ui(val, num);

	gmp_printf("%d! = %Zd\n", num, val);

	mpz_clear(val);
}
