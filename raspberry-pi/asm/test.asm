.section .data
  nl: .byte 10
  msg: .ascii "Hello World!"
  len = .-msg


.section .text
  .global _start

newline:
  mov r7, #4
  mov r0, #1
  ldr r1, nl_addr
  mov r2, #1
  swi 0x0

  mov pc, lr /* Return subroutine */

nl_addr: .word nl

_start:
  bl newline
  bl newline

  mov r7, #1
  mov r0, #0
  swi 0x0
