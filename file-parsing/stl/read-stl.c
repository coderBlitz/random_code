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
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include"stl.h"

int main(int argc, char *argv[]){
	if(argc != 2){
		fprintf(stderr, "Please provide a single filename.\n");
		return -1;
	}

	uint32_t triangle_cnt;
	struct triangle *triangles = stl_read_triangles_file(argv[1], &triangle_cnt);
	if(triangles == NULL){
		return -1;
	}

	printf("Count: %u\n", triangle_cnt);
	/*for(int i = 0;i < triangle_cnt;++i){
		printf("[%4d]: <%f, %f, %f>\n", i, triangles[i].norm.x, triangles[i].norm.y, triangles[i].norm.z);
	}*/

	free(triangles);

	return 0;
}
