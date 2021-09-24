.section .data
  prompt1: .ascii "Enter num 1: \0"
  len1 = .-prompt1
  prompt2: .ascii "Enter num 2: \0"
  len2 = .-prompt2
  test1: .int 0
  test2: .int 0

.section .text
.global _start

_start:
  mov $1,%rax # Write command
  mov $1,%rdi #Stdout stream
  mov $prompt1,%rsi #String
  mov $14,%rdx #Number of bytes
  syscall

  mov $0,%rax # Read call
  mov $0,%rdi # Stdin stream
  mov $test1,%rsi # Float variable
  mov $4,%rdx # 4-bytes for float
  syscall

  mov $1,%rax # Write command
  mov $1,%rdi #Stdout stream
  mov $prompt2,%rsi #String
  mov $len2,%rdx #Number of bytes
  syscall

  mov $0,%rax # Read call
  mov $0,%rdi # Stdin stream
  mov $test1,%rsi # Float variable
  mov $4,%rdx # 4-bytes for float
  syscall

/*  pushq %rbp
  movq %rsp,%rbp
  movl $test1,-8(%rbp) # Copy num1 to register
  movl $test2,-4(%rbp) # Add two numbers
  movl -8(%rbp),%edx
  movl -4(%rbp),%eax
  addl %edx,%eax
  movl %eax,-4(%rbp)
*/

  movq $test1,%r14
  movq $test2,%r15
  addq %r14,%r15

  mov $1,%rax # Write command
  mov $1,%rdi #Stdout stream
#  movl -4(%ebp),%esi #String
  mov %r15,%rsi
  mov $4,%rdx #Number of bytes
  syscall

#  popq %rbp

  mov $60,%rax #Exit call
  syscall
