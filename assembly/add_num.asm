# Register order:
# Syscall #|Param 1|Param 2|Param 3|Param 4|Param 5|Param 6
#   rax      rdi      rsi     rdx     r10     r8      r9

.section .bss
  .lcomm num1, 4
  .lcomm num2, 4

.section .text
 .global _start

_start:
  mov $0,%rax # Read function
  mov $0,%rdi # Stdin stream
  mov $num1,%rsi # First number variable
  mov $2,%rdx # 4-byte integer
  syscall

  sub $'0',(num1)

exit:
  mov $60,%rax # Exit call
  mov (num1),%rdi # Exit code
  syscall
