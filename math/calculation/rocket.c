#include<stdio.h>
#include<stdlib.h>

#define G 9.80665

static inline double Fd(double Cd, double Rho, double A, double v){
	return Cd * Rho * A * v*v / 2;
}

int main(int argc, char *argv[]){
	const double length = 0.311;
	const double diameter = 0.0241;
	const double A_body = 0.000506;
	const double Cd_body = 0.45;
	const double A_fins = 0.00496;
	const double Cd_fins = 0.01;
	const double mass_body = 0.0340;
	const double mass_eng_i = 0.0242;
	const double mass_eng_f = 0.0094;
	//const double rho = 1.2754; // At STP
	const double rho = 1.293;

	int iter = 1;
	const double dt = 0.1;
	double time = iter * dt;


	double mass = mass_body + mass_eng_i;

	const double t_cut = 1.8;
	const double mass_total_fuel = mass_eng_i - mass_eng_f;
	const double fuel_rate = mass_total_fuel / t_cut;
	double mass_fuel;

	const double thrust[] = {0, 6, 14, 5, 4.7, 4.4, 4.40625, 4.6, 4.5625, 4.5, 4.4425, 4.4875, 4.4425, 4.40625, 4.5, 4.4425, 4.4125, 4.4425, 4.4425};

	double force = thrust[1];
	double F_body, F_fins;

	double a = force/mass;
	double v = a*dt;
	double s = 0;

	// Print initial conditions
	//printf("S: %lf\tv: %lf\ta: %lf\n", s, v, a);
	printf("%4.2lf, %8.4lf, %8.4lf, %8.4lf, %8.4lf, %8.5lf\n", time, s, v, a, force, mass);

	time = ++iter * dt;

	// Thrust loop
	do{
		mass_fuel = mass_total_fuel - fuel_rate * time;
		mass = mass_body + mass_eng_f + mass_fuel;

		F_body = Fd(Cd_body, rho, A_body, v);
		F_fins = Fd(Cd_fins, rho, A_fins, v);

		force = thrust[iter] - F_body - F_fins - mass * G;

		a = force / mass;
		v += a*dt;
		s += v*dt;

		//printf("S: %lf\tv: %lf\ta: %lf\tF: %lf\n", s, v, a, force);
		printf("%4.2lf, %8.4lf, %8.4lf, %8.4lf, %8.4lf, %8.5lf\n", time, s, v, a, force, mass);
	}while((time = ++iter * dt) <= t_cut);

	// Change the things that remain constant
	mass_fuel = 0;
	mass = mass_body + mass_eng_f;
	//printf("End mass: %g\n", mass);

	// Coast loop
	do{
		F_body = Fd(Cd_body, rho, A_body, v);
		F_fins = Fd(Cd_fins, rho, A_fins, v);

		force = - F_body - F_fins - mass * G;

		a = force / mass;
		v += a*dt;
		s += v*dt;

		//printf("S: %lf\tv: %lf\ta: %lf\n", s, v, a);
		printf("%4.2lf, %8.4lf, %8.4lf, %8.4lf, %8.4lf, %8.5lf\n", time, s, v, a, force, mass);

		time = ++iter * dt;
	}while(v >= 0.0);
}
