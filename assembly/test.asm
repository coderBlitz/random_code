.section .text
 .global _start

_start:
  mov $len,%edx
  mov $msg,%ecx
  mov $1,%ebx # Where to write (STDOUT is 1)
  mov $4,%eax # Write sys call number
  int $0x80 # Kernel call

  movb $94,num; #movl-move long (presumably)
  mov $4,%eax
  mov $1,%ebx
  mov $num,%ecx
  mov $4,%edx
  int $0x80

  mov $len2,%edx
  mov $msg2,%ecx
  mov $1,%ebx
  mov $4,%eax
  int $0x80 # Kernel call

  mov $1,%eax # Exit
  mov $0,%ebx
  int $0x80

.section .data
 msg: .ascii "Hello, world! And all its inhabitants something something\n"
 len = .-msg
 msg2: .ascii "Hello ahain\n"
 len2 = .-msg2
.section .bss
 .lcomm num, 1
