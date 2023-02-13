#ifndef PNG_H
#define PNG_H

#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>

int parseFile(char *filename);

size_t PNGcreateChunk(const char *type, const void *data, size_t length, void *chunk_out);

uint32_t CRC32(uint32_t crc, const char *buf, const size_t len);

#endif
