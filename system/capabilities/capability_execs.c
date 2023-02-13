/* Testing capability behavior with exec calls
Notes:
	- The IAB set and other sets appear to conflict/overlap, causing issues.

	- Remaining root user keeps all capability sets unchanged
	- Downgrading EUID clears effective and ambient set
*/

#include<fcntl.h>
#include<linux/prctl.h>
#include<linux/securebits.h>
#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/capability.h>
#include<sys/prctl.h>
#include<sys/types.h>
#include<sys/wait.h>
#include<unistd.h>


bool has_cap(cap_value_t cap){
	bool ret = false;
	cap_t mycap = cap_get_proc();
	cap_flag_value_t cap_v;
	if(cap_get_flag(mycap, cap, CAP_EFFECTIVE, &cap_v) == -1){
		perror("Check cap error");
		cap_free(mycap);
		return false;
	}

	// If cap is set
	if(cap_v == CAP_SET){
		ret = true;
	}
	cap_free(mycap);

	return ret;
}

// Due to which sets are constrained by others, adds to permitted/interitable
//  first, then ambient, then effective.
bool add_cap(cap_value_t cap){
	cap_t caps = cap_get_proc();
	cap_value_t vals[] = {cap};
	int res;

	res = cap_set_flag(caps, CAP_PERMITTED, 1, vals, CAP_SET);
	if(res == -1){
		perror("Cap set permitted flag failed");
		return false;
	}
	res = cap_set_flag(caps, CAP_INHERITABLE, 1, vals, CAP_SET);
	if(res == -1){
		perror("Cap set inheritable flag failed");
		return false;
	}
	res = cap_set_proc(caps);
	if(res == -1){
		perror("Cap set proc failed");
		return false;
	}

	res = cap_set_ambient(cap, CAP_SET);
	if(res == -1){
		perror("set ambient failed");
		return false;
	}

	res = cap_set_flag(caps, CAP_EFFECTIVE, 1, vals, CAP_SET);
	if(res == -1){
		perror("Cap set effective flag failed");
		return false;
	}

	return true;
}

bool all_cap(void){
	int res;
	cap_t caps = cap_get_proc(); // XXX: Clearing caps  seems to prevent being able to set flags
	cap_value_t vals[] = {
		CAP_AUDIT_CONTROL,
		CAP_AUDIT_READ,
		CAP_AUDIT_WRITE,
		CAP_BLOCK_SUSPEND,
		CAP_BPF,
		CAP_CHECKPOINT_RESTORE,
		CAP_CHOWN,
		CAP_DAC_OVERRIDE,
		CAP_DAC_READ_SEARCH,
		CAP_FOWNER,
		CAP_FSETID,
		CAP_IPC_LOCK,
		CAP_IPC_OWNER,
		CAP_KILL,
		CAP_LEASE,
		CAP_LINUX_IMMUTABLE,
		CAP_MAC_ADMIN,
		CAP_MAC_OVERRIDE,
		CAP_MKNOD,
		CAP_NET_ADMIN,
		CAP_NET_BIND_SERVICE,
		CAP_NET_BROADCAST,
		CAP_NET_RAW,
		CAP_PERFMON,
		CAP_SETGID,
		CAP_SETFCAP,
		CAP_SETPCAP,
		CAP_SETUID,
		CAP_SYS_ADMIN,
		CAP_SYS_BOOT,
		CAP_SYS_CHROOT,
		CAP_SYS_MODULE,
		CAP_SYS_NICE,
		CAP_SYS_PACCT,
		CAP_SYS_PTRACE,
		CAP_SYS_RAWIO,
		CAP_SYS_RESOURCE,
		CAP_SYS_TIME,
		CAP_SYS_TTY_CONFIG,
		CAP_SYSLOG,
		CAP_WAKE_ALARM
	};
	int num_caps = sizeof(vals) / sizeof(*vals);
	//printf("Setting %d caps\n", num_caps);

	// Add capabilities to each primary set
	res = cap_set_flag(caps, CAP_PERMITTED, num_caps, vals, CAP_SET);
	if(res == -1){
		perror("Cap set permitted failed");
		return false;
	}
	res = cap_set_flag(caps, CAP_INHERITABLE, num_caps, vals, CAP_SET);
	if(res == -1){
		perror("Cap set inheritable failed");
		return false;
	}

	// Set process capabilities
	res = cap_set_proc(caps);
	if(res == -1){
		perror("Cap set proc failed");
		return false;
	}

	for(int i = 0;i < num_caps;i++){
		// Set ambient
		//res = cap_set_ambient(vals[i], CAP_SET);
		res = prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, vals[i], 0, 0);
		if(res == -1){
			perror("set ambient failed");
			return false;
		}
	}

	res = cap_set_flag(caps, CAP_EFFECTIVE, num_caps, vals, CAP_SET);
	if(res == -1){
		perror("Cap set effective failed");
		return false;
	}

	return true;
}


void exec_nodrop(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		// Do exec stuff
		execvp(cmd, args);

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

void exec_euid(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		int res = seteuid(65534); // Set EUID to nobody (shouldn't matter what this is, so long as it's valid, and not 0)
		if(res == -1){
			perror("UID change failed");
			exit(1);
		}
		setuid(65534);

		//add_cap(CAP_SYS_ADMIN);
		all_cap();

		// Do exec stuff
		execvp(cmd, args);

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

void exec_uid(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		int res = setuid(65534); // Set UID to nobody (shouldn't matter what this is, so long as it's valid, and not 0)
		if(res == -1){
			perror("UID change failed");
			exit(1);
		}

		//add_cap(CAP_SYS_ADMIN);
		all_cap();

		// Do exec stuff
		execvp(cmd, args);

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

void exec_uid_keep(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		prctl(PR_SET_SECUREBITS, SECBIT_KEEP_CAPS);

		int res = setuid(65534); // Set EUID to nobody (shouldn't matter what this is, so long as it's valid, and not 0)
		if(res == -1){
			perror("UID change failed");
			exit(1);
		}

		//add_cap(CAP_SYS_ADMIN);
		all_cap();

		// Do exec stuff
		execvp(cmd, args);

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

void exec_euid_nofix(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		prctl(PR_SET_SECUREBITS, SECBIT_NO_SETUID_FIXUP);

		int res = seteuid(65534); // Set EUID to nobody (shouldn't matter what this is, so long as it's valid, and not 0)
		if(res == -1){
			perror("UID change failed");
			exit(1);
		}

		//add_cap(CAP_SYS_ADMIN);
		all_cap();

		// Do exec stuff
		execvp(cmd, args);
		perror("Exec failed");

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

void exec_uid_nofix(char *restrict cmd,char **restrict args){
	pid_t child = fork();
	if(child == 0){
		prctl(PR_SET_SECUREBITS, SECBIT_NO_SETUID_FIXUP);

		int res = setuid(65534); // Set EUID to nobody (shouldn't matter what this is, so long as it's valid, and not 0)
		if(res == -1){
			perror("UID change failed");
			exit(1);
		}

		//add_cap(CAP_SYS_ADMIN);
		all_cap();

		// Do exec stuff
		execvp(cmd, args);

		exit(1);
	}else if(child == -1){
		perror("Fork failed");
	}

	// Wait for exit
	waitpid(child, NULL, 0);
}

int main(int argc, char *argv[]){
	// Command currently prints current cap info, then execs, sets some capability, and prints caps again
	//  to show behavior across multiple exec calls
	char *args[] = {"capsh", "--current", "--caps=cap_net_raw+eip", "--addamb=cap_net_raw", "--current", "==", "--current", "--caps=cap_net_raw+eip", "--current", NULL};

	all_cap();

	printf("------ Normal ------\n");
	exec_nodrop(args[0], args);

	printf("------ EUID change ------\n");
	exec_euid(args[0], args);

	printf("------ UID change ------\n");
	exec_uid(args[0], args);

	printf("------ UID change and keep_caps ------\n");
	exec_uid_keep(args[0], args);

	printf("------ EUID change and no_fixup ------\n");
	exec_euid_nofix(args[0], args);

	printf("------ UID change and no_fixup ------\n");
	exec_uid_nofix(args[0], args);

	// TODO: Chain execs with fixup, see if caps retain

	return 0;
}
