#include<endian.h>
#include<errno.h>
#include<fcntl.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/mman.h>
#include<sys/stat.h>
#include<unistd.h>

#ifndef CONTIGUOUS_FAT
#define CONTIGUOUS_FAT 0
#endif

#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define BLOCK_SIZE 512
#define SUPER_SIZE 64
#define BAK_SUPER_BLOCK 0
#define DIR_SIZE 32
#define FAT_SIZE 2 // Fat entry size, 2 bytes
#define DIR_ENTRIES_PER_BLOCK (BLOCK_SIZE / DIR_SIZE)
#define FAT_ENTRIES_PER_BLOCK (BLOCK_SIZE / FAT_SIZE)

#define BLOCK(a) ((a)*BLOCK_SIZE)

uint16_t SUPER_BLOCK = 1;
size_t NUM_BLOCKS = 0;


// MEMEFS superblock. Size = 64 bytes
struct meme_superb{
	int8_t signature[16]; // Signature (should be "?MEMEFS++CMSC421")
	uint8_t clean; // Clean unmount (0xFF on mount, clear on unmount)
	uint32_t version; // Version (set to 0x00000001)
	uint64_t creation; // Timestamp of FS creation (in Binary-Coded Decimal format)
	uint16_t fat_block; // Starting block of main FAT (set to 254 == 0x00FE)
	uint16_t fat_size; // main FAT size in blocks (set to 0x0001)
	uint16_t bak_fat_block; // Backup FAT starting block (set to 239 == 0x00EF)
	uint16_t bak_fat_size; // Backup FAT size in blocks (set to 0x0001)
	uint16_t dir_block; // Starting block of directory (set to 253 == 0x00FD)
	uint16_t dir_size; // Dir size in blocks (set to 14 == 0x000E)
	uint16_t user_size; // User blocks available (set to 220 == 0x00DC)
	uint16_t user_block; // Starting user block (set to 0x0001)
	int8_t label[16]; // Volume label (fill empty with NUL bytes)
};

// MEMEFS directory entry. Size = 32 bytes
struct meme_direntry{
	uint16_t perms; // 9 bit permissions, or 0 for unused
	uint16_t block; // Starting block
	int8_t name[12]; // Actually 11, but added 1 for null terminator here (implicitly a terminator is in direntry)
	uint64_t modified; // BCD timestamp of last modification
	uint32_t size; // File size in bytes
	uint16_t uid; // Owner UID
	uint16_t gid; // Owner GID
};

// Struct to encapsulate image struct data, to ease parameters required
struct meme_img{
	uint8_t *file;
	struct meme_superb *super;
	uint16_t *fat;
	struct meme_direntry *dirs;
	long fat_entries;
	long dirs_entries;
	long used_fat_entries;
};

/**	Parse primary superblock, from file_buf map, into struct given by sup
**/
void parse_super(uint8_t *img, struct meme_superb *sup, int8_t get_bak){
	if(img == NULL) return;

	const int64_t super_offset = (get_bak) ? 0 : BLOCK(SUPER_BLOCK);

	int8_t block[SUPER_SIZE]; // Buffer for superblock data
	memcpy(block, img + super_offset, SUPER_SIZE);

	memcpy(sup->signature, block + 0, 16);
	sup->clean = block[16]; // Offset 16 (17th byte)
	sup->version = be32toh(*(uint32_t *)(block + 20));
	sup->creation = be64toh(*(uint64_t *)(block + 24));
	sup->fat_block = be16toh(*(uint16_t *)(block + 32));
	sup->fat_size = be16toh(*(uint16_t *)(block + 34));
	sup->bak_fat_block = be16toh(*(uint16_t *)(block + 36));
	sup->bak_fat_size = be16toh(*(uint16_t *)(block + 38));
	sup->dir_block = be16toh(*(uint16_t *)(block + 40));
	sup->dir_size = be16toh(*(uint16_t *)(block + 42));
	sup->user_size = be16toh(*(uint16_t *)(block + 44));
	sup->user_block = be16toh(*(uint16_t *)(block + 46));
	memcpy(sup->label, block + 48, 16);
}

/**	Read the superblock into memory
**/
void read_super(struct meme_img *img, int8_t get_bak){
	if(img->file == NULL) return;

	static struct meme_superb sup; // Global scope
	img->super = &sup;

	parse_super(img->file, img->super, get_bak);
}

/**	Verify the default (or constrained) values
TODO: Check which other values are constrained (and to what)
**/
int verify_super_vals(struct meme_superb *sup){
	if(strcmp(sup->signature, "?MEMEFS++CMSC421")) return 1;
	if(sup->clean) return 1; // Check if clean unmount
	if(sup->version != 0x1) return 1;
	if(sup->fat_block == 0 && sup->fat_block >= SUPER_BLOCK) return 1; // First and last are always supers
	if(sup->fat_size == 0 || sup->fat_size > (NUM_BLOCKS / FAT_ENTRIES_PER_BLOCK)) return 1;
	if(sup->bak_fat_block == 0 && sup->bak_fat_block >= SUPER_BLOCK) return 1; // First and last are always supers
	if(sup->bak_fat_size == 0 || sup->bak_fat_size > 256) return 1;
	if(sup->dir_block == 0 && sup->dir_block >= SUPER_BLOCK) return 1; // First and last are always supers
	if(sup->dir_size == 0 || sup->dir_size >= SUPER_BLOCK) return 1;
	if(sup->user_block == 0 && sup->user_block >= SUPER_BLOCK) return 1; // First and last are always supers
	if(sup->user_size == 0 || sup->user_size >= SUPER_BLOCK) return 1;

	// Check creation timestamp (TODO: Do)
	uint8_t *date_fields = (uint8_t *)&sup->creation;

	// Verify block layouts (overlapping)
	int fat_a = sup->fat_block;
	int bfat_a = sup->bak_fat_block;
	int dir_a = sup->dir_block;
	int user_a = sup->user_block;
	int user_b = sup->user_block + sup->user_size;

	if(fat_a == dir_a) return 2; // If FAT and DIR start at the same block
	if(fat_a == bfat_a) return 2; // FAT and backup FAT cannot be same
	//if(user_a <= dir_a && user_b > dir_a) return 2; // If DIR starts in user block (NOTE: Clarified dir can be in user blocks)
	if(user_a <= fat_a && user_b > fat_a) return 2; // If FAT starts in user block

	// Check FAT position, only if more than 1 block; then FAT block must be within first FAT block.
	// First FAT must be able to read it's own next block, hence it must be within it's own range.
	if(sup->fat_size > 1){
		const int max_first = BLOCK_SIZE / FAT_SIZE;

		if(sup->fat_block >= max_first) return 3;
	}

	return 0;
}

/**	Given two superblocks, compare the two. Intended to compare main and backup blocks.
	Returns 0 on success, nonzero otherwise
**/
int verify_supers(struct meme_superb *main, struct meme_superb *bak){
	if(main == NULL || bak == NULL) return 0;

	if(verify_super_vals(main)) return 1;

	if(strncmp(main->signature, bak->signature, 16)) return 1;
	if(main->clean != bak->clean) return 1;
	if(main->version != bak->version) return 1;
	if(main->creation != bak->creation) return 1;
	if(main->fat_block != bak->fat_block) return 1;
	if(main->fat_size != bak->fat_size) return 1;
	if(main->bak_fat_block != bak->bak_fat_block) return 1;
	if(main->bak_fat_size != bak->bak_fat_size) return 1;
	if(main->dir_block != bak->dir_block) return 1;
	if(main->dir_size != bak->dir_size) return 1;
	if(main->user_size != bak->user_size) return 1;
	if(main->user_block != bak->user_block) return 1;
	if(strncmp(main->label, bak->label, 16)) return 1;

	return 0;
}

/**	Read FAT into buffer, up to N entries (all if N=0; ignored currently).
	Returns number of FAT entries.
	TODO: Add parameter to read back-up FAT table
**/
uint32_t read_fat(struct meme_img *img, uint32_t N, int8_t get_bak){
	if(img == NULL || img->file == NULL) return 0;

	uint16_t *fat = img->fat;
	const long total_blocks = (get_bak) ? img->super->bak_fat_size : img->super->fat_size;
	const long entries_per = FAT_ENTRIES_PER_BLOCK;

	// Get/alloc FAT array
	if(fat == NULL){
		fat = malloc(BLOCK_SIZE * total_blocks);
		if(fat == NULL) return 0;
		img->fat = fat;
	}

	//const long total_entries = entries_per * total_blocks;
	uint16_t block = (get_bak) ? img->super->bak_fat_block : img->super->fat_block;
	long block_cnt = 0;
	long entry_cnt = 0;
	long used_entries = 0;

	// TODO: Add a condition to check for possible chain (if fat_size > 1 and fat[block] is not FFFF)
#if CONTIGUOUS_FAT
	// Loop over FAT entries (assumes FAT is chain; NOTE: 421 project is not a chained FAT)
	do{
		// Copy the FAT block
		memcpy(fat + entry_cnt, img->file + BLOCK(block), BLOCK_SIZE);

		// Change endianness of entries
		for(long i = 0;i < entries_per;i++){
			if(fat[entry_cnt + i]) used_entries += 1; // If FAT entry nonzero, used entry

			fat[entry_cnt + i] = be16toh(fat[entry_cnt + i]);
		}
		entry_cnt += entries_per; // Add entries counted

		block = fat[block]; // FAT *MUST* be within first FAT block
	}while(++block_cnt < total_blocks && block != 0xFFFF);
#else
	// Copy the FAT block
	memcpy(fat, img->file + BLOCK(block), total_blocks * BLOCK_SIZE);

	// Change endianness of entries
	for(long i = 0;i < total_blocks * entries_per;i++){
		if(fat[i]) used_entries += 1; // If FAT entry nonzero, used entry

		fat[i] = be16toh(fat[i]);
	}
	entry_cnt = total_blocks * entries_per;
#endif

	img->fat_entries = entry_cnt; // Export counts
	img->used_fat_entries = used_entries;

	return entry_cnt;
}

/**	Test FAT entries to known values.
	TODO: Modify to be "verify_fat_entry" (or make another fn), given starting block and expected chain length
		TODO: Only cycle detect the given block.
	TODO: With above modification, this should only check superblock stuff to be 0xFFFF.
**/
int verify_fat_entries(struct meme_img *img, char bak){
	if(img == NULL || img->fat == NULL || img->super == NULL) return 1;

	/* Checks
		TODO: Cycle detection for every entry (Brent's algo). Cycle length 1 with start 0xFFFF is valid, else invalid
		TODO: Validate reported DIR and FAT lengths (FAT length <= 256 by design; only if FAT is chained)
		TODO: Validate user/dir blocks remain in user/dir blocks
	*/
	size_t total_entries = ((bak) ? img->super->fat_size : img->super->bak_fat_size) * FAT_ENTRIES_PER_BLOCK - 1;
	size_t MAX_CHAIN = total_entries; // May not be always total entries, since some blocks are reserved
	uint16_t *fat = img->fat;

	// Check superblock entries (must be 0xFFFF)
	if(fat[0] != 0xFFFF) return 2;
	if(fat[SUPER_BLOCK] != 0xFFFF) return 2;

	// Verify all entries are in range
	for(size_t i = 1;i < total_entries;i++){
		if(fat[i] == 0xFFFF || fat[i] == 0x0000) continue; // End-of-chain marker, or zero (unused), completely valid
		if(fat[i] >= total_entries) return 3; // Verify all fat blocks are within range
	}

	// Do loop detection
	for(size_t i = 1;i < total_entries;i++){
		if(fat[i] == 0xFFFF || fat[i] == 0x0000) continue;

		/* Find lambda and mu such that x_{mu} = x_{lambda + mu} (the cycle)
		First find lambda (sequence length), then move hare to lambda, then move both to find mu
		*/
		size_t p = 1, lam = 1;
		uint16_t t = fat[i]; // Starting block
		uint16_t h = fat[t]; // Next block
		while(t != h){
			if(lam == p){
				p *= 2; // Increment p in any way (exponential best)
				lam = 0;
				t = h;
			}

			h = fat[h];
			lam += 1;
		}

		// Separate hare and turtle by lambda
		t = fat[i];
		h = fat[i];
		for(size_t j = 0;j < lam;++j){
			h = fat[h];
		}

		// Find mu (start) of sequence
		uint64_t mu = 0;
		while(t != h){
			t = fat[t];
			h = fat[h];
			++mu;
		}

		if(t != 0xFFFF){
			fprintf(stderr, "ERROR: Cycle detected!\n");
			fprintf(stderr, "\tBlock 0x%04hX\n", i);
			fprintf(stderr, "\tMu = %8lu\tLambda = %8lu\n", mu, lam);
			return 4;
		}
	}

	return 0;
}

/**	Util function for read_dirs
**/
void extract_direntry(uint8_t *buffer, struct meme_direntry *dir){
	if(buffer == NULL || dir == NULL) return;

	dir->perms = be16toh(buffer[0x0]);
	dir->block = be16toh(buffer[0x2]);
	memcpy(dir->name, buffer + 0x4, 11);
	dir->name[0xC] = 0;
	dir->modified = be64toh(buffer[0x10]);
	dir->size = be32toh(buffer[0x18]);
	dir->uid = be16toh(buffer[0x1C]);
	dir->gid = be16toh(buffer[0x1E]);
}

/**	Read up to N dir entries (all if N=0) into struct array parameter, starting
	 from 'off' entry.
	Returns number of entries read, which is (mostly) based on total blocks read.
**/
uint32_t read_dirs(struct meme_img *img, uint32_t N, uint32_t off){
	if(img == NULL || img->file == NULL) return 0;
	if(N == 0) N = img->super->dir_size * BLOCK_SIZE / DIR_SIZE; // N becomes max entries

	const uint16_t dir_size = img->super->dir_size; // Block count
	const uint16_t dirs_per_block = BLOCK_SIZE / DIR_SIZE;
	uint8_t buffer[BLOCK_SIZE];
	//printf("Dirs per: %hu\n", dirs_per_block);

	// If directory non-existent, or offset beyond dir size, exit early
	if(!dir_size || off > (dirs_per_block * dir_size)) return 0;

	// Fetch FAT if necessary
	if(img->fat == NULL){
		read_fat(img, 0, 0);
	}

	// Allocate DIRS array if necessary
	struct meme_direntry *entries = NULL;
	if(img->dirs == NULL){
		entries = malloc(N * sizeof(*entries));
		img->dirs = entries;
	}
	entries = img->dirs;

	uint16_t start_idx = off / dirs_per_block;
	off %= dirs_per_block; // Get first index inside DIR where offset starts

	// Iterate to starting block
	uint16_t block = img->super->dir_block;
	uint32_t idx = 0;
	while(idx < start_idx && block != 0xFFFF){
		//printf("Skipping block %2d..\n", block);
		//printf("Next block = 0x%04hX\n", img->fat[block]);

		block = img->fat[block];
		++idx;
	}

	struct meme_direntry dir;
	uint32_t cnt = 0; // Counter for dirs array
	uint32_t read_amt = MIN(N, dirs_per_block - off); // Block size, or that minus
	//printf("Amt = %4u\n", read_amt);

	// Get first block set, with possible offset
	memcpy(buffer, img->file + BLOCK(block) + off, read_amt * DIR_SIZE);

	// Parse all entries into array
	for(long i = 0;i < (dirs_per_block-off);i++){
		extract_direntry(buffer + i * DIR_SIZE, &dir);

		entries[cnt++] = dir;
	}
	block = img->fat[block]; // Get next block

	// While directory blocks still exist
	while(++idx < dir_size && block != 0xFFFF){
		// Get read amounts
		N -= read_amt;
		read_amt =  MIN(N, dirs_per_block);
		//printf("Amt = %4u\n", read_amt);

		// Copy then parse buffer
		memcpy(buffer, img->file + BLOCK(block), read_amt * DIR_SIZE);
		for(long i = 0;i < (dirs_per_block-off);i++){
			extract_direntry(buffer + i * DIR_SIZE, &dir);

			entries[cnt++] = dir; // Append to entries array
		}

		block = img->fat[block];
	}
	img->dirs_entries = cnt;

	return cnt;
}

/**	Get all vital information from mapped file, store in memory.
**/
int read_all(struct meme_img *img){
	if(img == NULL || img->file == NULL) return 1;

	// Get and verify supers
	read_super(img, 0);
	if(verify_super_vals(img->super)){
		fprintf(stderr, "WARNING: Primary superblock has bad entry/entries!\n");
		read_super(img, 1); // Get backup block
		if(verify_super_vals(img->super)){
			fprintf(stderr, "ERROR: Both superblocks have bad entry/entries!\n");
			// TODO: (maybe; not for 421 proj) Attempt to fix whatever values can be fixed
			return 1; // Don't continue if both superblocks are bad
		}
		// TODO: Fix main superblock (write backup values to main)
		//write_supers(img);
	}

	// Get and verify FATs
	read_fat(img, 0, 0);
	if(verify_fat_entries(img, 0)){
		fprintf(stderr, "WARNING: Primary FAT bad entry/entries!\n");
		read_fat(img, 0, 1); // Get backup
		if(verify_fat_entries(img, 1)){
			fprintf(stderr, "ERROR: Both FATs have bad values\n");
			// TODO: Fix if possible (set to 0xFFFF for known blocks, 0x0000 otherwise)
			return 1;
		}
		// TODO: Fix main FAT (copy backup value to main)
	}

	// Get and verify all dirs
	read_dirs(img, img->super->dir_size * DIR_ENTRIES_PER_BLOCK, 0);

	return 0;
}

int main(int argc, char *argv[]){
	if(argc < 2){
		printf("Please specify an image to parse.\n");
		return EINVAL;
	}

	int res = 0;

	char *file = argv[1];
	printf("Parsing '%s'\n", file);

	int img = open(file, O_RDONLY);
	if(img < 0){
		perror("Could not open image");
		return errno;
	}

	// Grab file size
	struct stat sb;
	fstat(img, &sb);
	long num_blocks = sb.st_size / BLOCK_SIZE;
	printf("Image size: %8lu\tBlocks: %6lu\n", sb.st_size, num_blocks);

	if(num_blocks < 2){
		fprintf(stderr, "ERROR: Image too small\n");
		return 2;
	}else if(num_blocks > (256*256)){
		fprintf(stderr, "ERROR: Image too large\n");
		// TODO: Optionally check 0xFFFF and 0x0000 for valid superblocks, then use/fix
		return 2;
	}

	SUPER_BLOCK = num_blocks - 1; // Set super block to last block (if not constant) TODO: Change depending on above error(s)
	NUM_BLOCKS = num_blocks;
	printf("Setting superblock to %6hu\n", SUPER_BLOCK);

	// Map to buffer
	char *img_buf = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, img, 0);
	if(img_buf == MAP_FAILED){
		perror("Could not map file");
		return errno;
	}
	close(img); // No longer needed

	// IMG struct
	struct meme_img cur_img = {img_buf, NULL, NULL, NULL};

	/******* Manual extraction of stuff *******
	// Parse image parts
	read_super(&cur_img, 0);
	struct meme_superb sup = *cur_img.super;
	printf("Image signature: '%.16s'\n", sup.signature);
	printf("Clean: 0x%02hhX\n", sup.clean);
	printf("Version: 0x%08X\n", sup.version);
	printf("Creation: 0x%016lX\n", sup.creation);
	printf("FAT start: 0x%04hX\n", sup.fat_block);
	printf("FAT size: 0x%04hX\n", sup.fat_size);
	printf("BAK FAT start: 0x%04hX\n", sup.bak_fat_block);
	printf("BAK FAT size: 0x%04hX\n", sup.bak_fat_size);
	printf("Dir start: 0x%04hX\n", sup.dir_block);
	printf("Dir size: 0x%04hX\n", sup.dir_size);
	printf("User start: 0x%04hX\n", sup.user_block);
	printf("User size: 0x%04hX\n", sup.user_size);
	printf("Image label: '%.16s'\n", sup.label);

	// Fetch back-up, then verify primary and back-up match
	struct meme_superb bak;
	parse_super(img_buf, &bak, 1);

	res = verify_supers(&sup, &bak);
	if(!res){
		printf("Superblocks match and are valid!\n");
	}else{
		fprintf(stderr, "WARNING: Superblocks DO NOT match OR have problems!\n");
		return 1;
	}

	// Copy the fat block(s). 80% confident they must be contiguous (TODO: Verify)
	long total_entries = read_fat(&cur_img, 0, 0);
	uint16_t *fat = cur_img.fat;
	printf("Total FAT entries = %ld\n", total_entries);
	for(long i = 0;i < total_entries;i++){
		// Print all non-zero FAT entries
		if(fat[i]){
			printf("[0x%016lX] = 0x%04hX\n", i, fat[i]);
		}
	}
	if(!total_entries){
		fprintf(stderr, "Parsed empty FAT!\n");
		return 2;
	}

	// Allocate directory array
	long num_dirs = sup.dir_size * BLOCK_SIZE / DIR_SIZE;

	// Grab directory entries
	printf("Fetching dirs..\n");
	num_dirs = read_dirs(&cur_img, 0, 0);
	struct meme_direntry *entries = cur_img.dirs;
	printf("Got %3lu dirs\n", num_dirs);

	char *ext;
	for(long i = 0;i < num_dirs;i++){
		if(entries[i].perms){
			ext = entries[i].name + 0x8;
			printf("File: '%.8s.%.3s'\n", entries[i].name, ext);
		}
	}
	*****************************/

	// Read all things from file
	res = read_all(&cur_img);
	if(!res){
		printf("At least one superblock is valid!\n\n"); // TODO: Update once read_all fixes super/FAT
	}else{
		fprintf(stderr, "ERROR: Superblocks DO NOT match OR have problems!\n");
		return 1;
	}

	// Extract individual entries
	struct meme_superb sup = *cur_img.super;
	uint16_t *fat = cur_img.fat;
	struct meme_direntry *entries = cur_img.dirs;
	long fat_entries = cur_img.fat_entries;
	long used_entries = cur_img.used_fat_entries;
	long dirs_entries = cur_img.dirs_entries;

	// Print super info
	printf("Image signature: '%.16s'\n", sup.signature);
	printf("Clean: 0x%02hhX\n", sup.clean);
	printf("Version: 0x%08X\n", sup.version);
	printf("Creation: 0x%016lX\n", sup.creation);
	printf("FAT start: 0x%04hX\n", sup.fat_block);
	printf("FAT size: 0x%04hX\n", sup.fat_size);
	printf("BAK FAT start: 0x%04hX\n", sup.bak_fat_block);
	printf("BAK FAT size: 0x%04hX\n", sup.bak_fat_size);
	printf("Dir start: 0x%04hX\n", sup.dir_block);
	printf("Dir size: 0x%04hX\n", sup.dir_size);
	printf("User start: 0x%04hX\n", sup.user_block);
	printf("User size: 0x%04hX\n", sup.user_size);
	printf("Image label: '%.16s'\n", sup.label);

	// Print non-zero FAT table entries
	printf("Total FAT entries = %5ld\tUsed = %5ld\n", fat_entries, used_entries);
	if(fat_entries <= 512){
		for(long i = 0;i < fat_entries;i++){
			// Print all non-zero FAT entries
			if(fat[i]){
				printf("[0x%016lX] = 0x%04hX\n", i, fat[i]);
			}
		}
	}else{
		printf("FAT too large to print.\n");
	}

	// Print non-empty directory entries
	printf("Got %3lu dirs\n", dirs_entries);
	char *ext;
	for(long i = 0;i < dirs_entries;i++){
		if(entries[i].perms){
			ext = entries[i].name + 0x8;
			printf("File: '%.8s.%.3s'\n", entries[i].name, ext);
		}
	}


	// Clean up
	free(entries);
	free(fat);
	munmap(img_buf, sb.st_size);

	return 0;
}
