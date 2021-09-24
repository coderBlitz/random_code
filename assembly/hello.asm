.section .data
  message: .ascii "Hello world!\n\0" # Hello world with newline
  len = .-message

.section .text
.global _start

_start:
  movl $4,%eax #Write syscall
  movl $1,%ebx #stdout is stream 1
  movl $message,%ecx #Put message to write
  movl $len,%edx #How many bytes to write
  int $0x80 #Kernel call

  mov $60,%rax #Exit syscall
  mov $29,%rdi
  syscall
# x64 exit code is 60
