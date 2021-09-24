.SECTION .data
	N: .quad 10000000		# N is the highest number to go to
	nl: .byte 10
	colon: .ascii ": "
	final_string: .ascii "Largest chain length: "
	final_string_len = .-final_string

	even: .ascii "even\n"
	even_len = .-even
	odd: .ascii "odd\n"
	odd_len = .-odd
.SECTION .bss
.SECTION .text
.global _start

.extern prt_dec

newline:
	pushq %rax
	pushq %rbx
	pushq %rcx
	pushq %rdx
#	pushq %rdi

	movq $1, %rax	# Write
	movq $1, %rdi	# Stdout
	movq $nl, %rsi	# Write from nl
	movq $1, %rdx	# Write 1 byte
	syscall

#	popq %rdi
	popq %rdx
	popq %rcx
	popq %rbx
	popq %rax
ret

# Takes argument from stack, returns the number of hops for said argument
collatz:
	pushq %rbp
	movq %rsp, %rbp

	pushq %r12
	pushq %r13

	movq 16(%rbp), %r12	# r12 will store the number
#	cmp $1, %r12				# Check if 1 or less
#	jle .end_collatz
	
	xor %r13, %r13				# r13 will store the count

.loop_collatz:
	movq $1, %rbx
	and %r12, %rbx			# Determine whether even or odd (rbx value of 1 is odd)
	inc %r13

	cmp $0, %rbx
	je .even
.odd:
#	movq $1, %rax	# Write
#	movq $1, %rdi	# Stdout
#	movq $odd, %rsi	# Write from nl
#	movq $odd_len, %rdx	# Write 1 byte
#	syscall

	imul $3, %r12
	add $1, %r12

	jmp .loop_collatz
.even:
#	movq $1, %rax	# Write
#	movq $1, %rdi	# Stdout
#	movq $even, %rsi	# Write from nl
#	movq $even_len, %rdx	# Write 1 byte
#	syscall

	shr $1, %r12

	cmp $1, %r12
	jne .loop_collatz

.end_collatz:
	mov %r13, %rax			# Store return

	popq %r13
	popq %r12

	movq %rbp, %rsp
	popq %rbp

ret

#	START/MAIN
_start:
	movq $2, %r12		# r12 is the counter
	movq N, %r13		# r13 is the end number
	xor %r14, %r14		# Zero these out
	xor %r15, %r15
.loop_start:
#  	Calculate chain length
	pushq %r12
	call collatz
	popq %r12

	cmp %rax, %r14
	jge .incr
	movq %rax, %r14				# Store result in r14 only if greater
	movq %r12, %r15				# Store the number with highest chain

#	Print the number
#	call prt_dec

#	Print a colon after the number
#	movq $1, %rax
#	movq $1, %rdi
#	movq $colon, %rsi
#	movq $2, %rdx
#	syscall

#	Print the chain length
#	pushq %r14
#	call prt_dec
#	call newline

.incr:
	inc %r12
	cmp %r13, %r12
	jle .loop_start

#	Print final details
	movq $1, %rax
	movq $1, %rdi
	movq $final_string, %rsi
	movq $final_string_len, %rdx
	syscall

	pushq %r14
	call prt_dec
	call newline

#	Move stack pointer back
	pushq %r15
	call prt_dec
	call newline

#	Exit
	movq $60, %rax
	movq $0, %rdi
	syscall
