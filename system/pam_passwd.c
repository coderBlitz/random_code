#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <security/pam_client.h>
#include <security/pam_appl.h>

int empty_conv(int num_msg, const struct pam_message **msg, struct pam_response **resp, void *data) {
		// Allocate the array of responses.
		*resp = malloc(num_msg * sizeof(**resp));
		if (*resp == NULL) {
				return PAM_BUF_ERR;
		}

		// For every message give a response with empty string.
		for(int i = 0;i < num_msg;i++) {
				printf("Message[%d]: %s\n", i, msg[i]->msg);

				(*resp)[i].resp_retcode = 0;

				// Allocate response string, and null-terminate it.
				if (msg[i]->msg_style == PAM_PROMPT_ECHO_OFF
						|| msg[i]->msg_style == PAM_PROMPT_ECHO_ON) {
						char *new_passwd = "ubuntu";
						(*resp)[i].resp = malloc(strlen(new_passwd) + 1);
						strcpy((*resp)[i].resp, new_passwd);
				} else {
						(*resp)[i].resp = malloc(1);
						(*resp)[i].resp[0] = '\0';
				}
		}

		return PAM_SUCCESS;
}

int main() {
		pam_handle_t *hand;
		struct pam_conv empty = { empty_conv, NULL };
		int res = pam_start("passwd", "ubuntu", &empty, &hand);
		if(res != PAM_SUCCESS) {
				printf("pam_start failed: %d\n", res);
				return res;
		}

		// Try to change password.
		res = pam_chauthtok(hand, 0);
		if(res != PAM_SUCCESS) {
				printf("Failed to change auth token.\n");
		}

		pam_end(hand, res);
}
