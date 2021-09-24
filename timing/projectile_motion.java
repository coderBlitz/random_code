import java.util.Scanner;

public class projectile_motion{
	public static void main(String[] args){
		System.out.printf("Ideal Projectile Simulation v1.0\n");


		float START_Y_POS = 0.0f;// Height at which object is launched from
		float INIT_VELOCITY = 0.0f;// Velocity in generic measurement (distance unit/second)
		float INIT_ANGLE = 0.0f;// Launch angle in degrees

		// See if 'i' or '-i' is given to allow manual entry
		try{
		if(args[0].charAt(0) == 'i' || args[0].charAt(1) == 'i'){
			Scanner s = new Scanner(System.in);
			System.out.printf("Enter starting height: ");
			START_Y_POS = s.nextFloat();
			System.out.printf("Enter Initial Velocity: ");
			INIT_VELOCITY = s.nextFloat();
			System.out.printf("Enter Firing angle: ");
			INIT_ANGLE = s.nextFloat();
			s.close();
		}
		}catch(Exception e){
			START_Y_POS = 10.0f;// Height at which object is launched from
			INIT_VELOCITY = 15.0f;// Velocity in generic measurement (distance unit/second)
			INIT_ANGLE = 20.0f;// Launch angle in degrees
		}

		float x_pos=0.0f, y_pos = START_Y_POS;// Will be current stats of projectile
		float X_VELOCITY = INIT_VELOCITY * (float)Math.cos(INIT_ANGLE * Math.PI/180.0);// Horizontal speed
		float Y_VELOCITY = INIT_VELOCITY * (float)Math.sin(INIT_ANGLE * Math.PI/180.0);// Vertical speed

		System.out.printf("INITIAL HEIGHT: %.03f\n",START_Y_POS);
		System.out.printf("INITIAL SPEED: %.03f\n",INIT_VELOCITY);
		System.out.printf("ANGLE: %.03f degrees\n",INIT_ANGLE);
		System.out.printf("LAUNCH!\n");

		float cur_time;
		float start = System.nanoTime()/10e8f;// Get launch/start time
		while(y_pos > 0){// While projectile is still in flight
			cur_time = System.nanoTime()/10e8f - start;// Make time easier for calculations

			x_pos = cur_time * X_VELOCITY;
			y_pos = START_Y_POS + Y_VELOCITY * cur_time +(0.5f * -9.8f * cur_time*cur_time);// Get height
			// y = init_vert_velocity * time + 0.5 * vert.accel * time^2
			// vert.accel AKA Gravity
			// Gravity is -9.8 m/s/s

			System.out.printf("\rTime: %.05f\t X_POS: %.06f\tY_POS: %.06f",cur_time,x_pos,y_pos);
		}
		cur_time = System.nanoTime()/10e8f - start;

	System.out.printf("\nLANDED\nProjectile went %.04f distance units in %.06f seconds.\n",x_pos,cur_time);
	}
}
