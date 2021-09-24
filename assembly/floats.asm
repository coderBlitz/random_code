.section .data

.section .bss

.section .text
 .global add

add:
#  pushq %rbp
#  movq %rsp, %rbp

  addps %xmm1, %xmm0

#  movq %rbp, %rsp # 'leave' should replace these 3 lines
#  popq %rbp
  ret
