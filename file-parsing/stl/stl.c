#include"stl.h"

struct triangle *stl_read_triangles_file(char *filename, uint32_t *count){
	FILE *fp = fopen(filename, "rb");
	if(fp == NULL) return NULL;

	struct triangle *res = stl_read_triangles(fp, count);

	fclose(fp);

	return res;
}

struct triangle *stl_read_triangles(FILE *fp, uint32_t *count){
	char head[80];
	uint32_t triangle_count;

	*count = 0;

	fread(head, 1, 80, fp);
	fread(&triangle_count, 4, 1, fp);

	//printf("Header: %s\n", head);
	//printf("Count: %u\n", triangle_count);

	struct triangle *triangles = malloc(triangle_count * sizeof(*triangles));
	if(triangles == NULL){
		fprintf(stderr, "Triangle alloc failed.\n");
		return NULL;
	}

	unsigned char buffer[48];
	int res;
	for(uint32_t i = 0;i < triangle_count;++i){
		struct vec3 normal;
		struct vec3 v1, v2, v3;
		uint16_t att_cnt;

		res = fread(buffer, 4, 12, fp); // Read normal & XYZ data
		res += fread(&att_cnt, 2, 1, fp); // Read attribute byte count
		if(res != 13){
			fprintf(stderr, "Read error.\n");
			free(triangles);
			return NULL;
		}

		float *points = (float *) buffer;
		normal.x = points[0];
		normal.y = points[1];
		normal.z = points[2];

		v1.x = points[3];
		v1.y = points[4];
		v1.z = points[5];

		v2.x = points[6];
		v2.y = points[7];
		v2.z = points[8];

		v3.x = points[9];
		v3.y = points[10];
		v3.z = points[11];

		triangles[i].norm = normal;
		triangles[i].verts[0] = v1;
		triangles[i].verts[1] = v2;
		triangles[i].verts[2] = v3;

		/*printf("Triangle %5u{\n", i);
		printf("  <%f, %f, %f>\n", normal.x, normal.y, normal.z);
		printf("  (%f, %f, %f)\n", v1.x, v1.y, v1.z);
		printf("  (%f, %f, %f)\n", v2.x, v2.y, v2.z);
		printf("  (%f, %f, %f)\n", v3.x, v3.y, v3.z);
		printf("  Attributes: %hu\n", att_cnt);
		printf("}\n");*/

		fseek(fp, att_cnt, SEEK_CUR);
	}

	*count = triangle_count;
	return triangles;
}
