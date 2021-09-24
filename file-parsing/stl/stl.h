/*	Read a binary STL file
STL format:
	80-byte header
	4-byte unsigned integer [little-endian] of triangle count

	triangle{
		Normal: 3x float32
		Vertices: 3x pairs of XYZ coordinates (9x total float32)
		Attribute byte count: 2-byte unsigned integer
		Attributes
	}
*/

#ifndef STL_H
#define STL_H

#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

struct vec3{
	float x;
	float y;
	float z;
};

struct triangle{
	struct vec3 verts[3]; // XYZ order
	struct vec3 norm;
};

struct triangle *stl_read_triangles_file(char *, uint32_t *);
struct triangle *stl_read_triangles(FILE *, uint32_t *);

#endif
