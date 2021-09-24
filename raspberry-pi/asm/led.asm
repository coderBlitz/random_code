.section .data

.section .text
	.global _start

_start:
/*	ldr r0, =0x20200000 /* Base address */
	ldr r0, =0x3F200000 /* Could also be base address */

	/* Set to input */
	mov r2, #7 /* Bitmask */
	lsl r2, #18 /* Shift */
	mvn r1, r2 /* Inverting 7 gives all zeroes */
	and r1, r0
	str r1, [r0]

	/* Then output */
	mov r1, #1 /* Set 1 */
	lsl r1, #18 /* Shift */
	str r1, [r0]

	/* Then set GPIO */
	mov r1, #1
	lsl r1, #6
	str r1, [r0,#28]
