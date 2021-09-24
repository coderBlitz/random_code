.section .text
 .global _start

newline:
  pushq %rax
  pushq %rbx
  pushq %rcx
  pushq %rdx

#  mov $4, %eax
#  mov $1, %ebx
#  mov $nl, %ecx
#  mov $1, %edx
#  int $0x80
  mov $SYSCALL_WRITE, %rax
  mov $1, %rdi
  mov $nl, %rsi
  mov $1, %rdx
  syscall

  popq %rdx
  popq %rcx
  popq %rbx
  popq %rax
  ret

_start:
  mov $len,%edx
  mov $msg,%ecx
  mov $1,%ebx # Where to write (STDOUT is 1)
  mov $4,%eax # Write sys call number
  int $0x80 # Kernel call

  movb $94,(num); #movl-move long (presumably)
  movb $11,(num2)
  movb $11, (4+num2)

  movl $10000, (0+nums)
  movl $20000, (4+nums)
  movl $30000, (8+nums)
  movl $40000, (12+nums)

xmm:
  movq (num), %xmm0
  movaps (nums), %xmm2

#  movq (num2), %xmm1 # Equivalent to movlps
  movlps (num2), %xmm1
  movhps (num2), %xmm1 # Load memory value into upper 64 bits of register
  divps %xmm1, %xmm2
  cvtps2pi %xmm2, %mm0 # Convert single precision to integer
  movq %mm0, nums # Store integers

  movq %xmm2, %xmm3 # Swap upper and lower then repeat conversion
  movhlps %xmm2, %xmm2
  movlhps %xmm3, %xmm2

  cvtps2pi %xmm2, %mm0 # Convert second set
  movq %mm0, (nums+8) # Store second set

  movlhps %xmm1, %xmm0 # Moves lower 64-bits of arg1 to upper 64-bits of arg2
  addps %xmm1, %xmm0 # Addss only adds lower 32 bits (ps does all 4 32 bit sets)
  movss %xmm0, (num)

  mov $4,%eax
  mov $1,%ebx
  movl $num,%ecx
  mov $4,%edx
  int $0x80

  call newline

  mov $len2,%edx
  mov $msg2,%ecx
  mov $1,%ebx
  mov $4,%eax
  int $0x80 # Kernel call

  mov $1,%eax # Exit
  mov $0,%ebx
  int $0x80

.section .data
 SYSCALL_WRITE = 1
 msg: .ascii "Hello, world! And all its inhabitants something something\n"
 len = .-msg
 msg2: .ascii "Hello ahain\n"
 len2 = .-msg2
 nl: .ascii "\n"
.section .bss
 .lcomm num, 8
 .lcomm num2, 8

 .align 16 # "Required" for loading 128 bits into XMM registers (might complain)
 .lcomm nums, 16 # Align affects this variable
