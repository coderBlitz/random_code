# Compile using GAS, link using GCC

.section .text
	.global main

main:
	movq $8, %rdi
	call malloc

	movq $123, (%rax)

	mov %rax, %rdi
	call free

	ret
