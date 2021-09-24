.equ STDOUT, 1
.equ SYSCALL_WRITE, 4

.section .data
  msg: .ascii "Adding floats..\n"
  len = .-msg

.section .text
  .global test

test:
  mov r12, sp
  push {r4,r7}

  push {r0,r1}

  mov r7, #SYSCALL_WRITE
  mov r0, #STDOUT
  ldr r1, msg_addr
  mov r2, #1
  swi 0x0 /* Software-Interrupt, basically the kernel call */

/*  fadds s0,s0, s1 */
  pop {r0,r1}
  add r0,r0,r1

  fmsr s0, r0
  fsitos s1, s0

  pop {r4,r7}


  mov pc, lr /* End/Return */

msg_addr: .word msg /* ARM necessity due to data and text separation */
len_addr: .word len
