.set SYSCALL_WRITE, 4
.set STDOUT, 1
.set STRING_LENGTH, 20


.SECTION .bss
  .lcomm numString STRING_LENGTH # Hold digits to print
  .lcomm numDigits 1

.SECTION .text
.global prt_dec
prt_dec:
	pushq %rax # Save all registers
	pushq %rbx
	pushq %rcx
	pushq %rdx

	movq 40(%rsp), %rax # Get number (8 bytes * (1 argument + 4 registers) = 40 bytes)

	movl $0, %ecx
.clear_loop:
	cmp $STRING_LENGTH, %ecx
	je .clear_done

	movb $0, numString(%ecx) # Clear string digits

	inc %ecx
	jmp .clear_loop

.clear_done:

# Algorithm:
# eax - Value
# ebx - hold 10 for division instruction
# ecx - digit counter
# edx - remainder
# Loop
# - Compare eax to 10
# - If less than: push eax to stack, then finish
# - else: divide eax by 10 and push edx to stack, then jump to loop
# - finish: Using counter and string variable, pop all digits from stack into string
# - Then print string
	movl $10, %ebx # Hold divisor
	movl $0, %ecx  # Clear counter
.loop:
	cmp $10, %eax          # Less than 10 means last digit
	jae .continue         # Deal with 2 or more digits (unsigned)
	inc %ecx              # Add last count
	mov %ecx, (numDigits) # Store digit count in variable
	pushq %rax             # Put last digit on stack

	movl %ecx, %ebx # Use ebx to retain digit count
	movl $0, %ecx   # Use ecx to count up in string
	jmp .done    # Make and print string

.continue:
	movl $0, %edx # Clear dividend (upper-half)
	div %ebx    # Divide eax/ebx (eax/10), eax has quotient, remainder in edx

	pushq %rdx  # Push remainder digit
	inc %ecx   # Increase digit count
	jmp .loop # Repeat

.done:
	cmp %ebx, %ecx
	je .finish

	popq %rax # Get first
	add $'0', %eax
	movl %eax, numString(%ecx) # Move digit in eax to the string

	inc %ecx
	jmp .done # Repeat for every digit

.finish:
	movl $SYSCALL_WRITE, %eax
	movl $STDOUT, %ebx
	movl $numString, %ecx
	movl (numDigits), %edx
	int $0x80

	popq %rdx # Restore all registers
	popq %rcx
	popq %rbx
	popq %rax
	ret
