use std::env;
use std::fs;
use std::io::{Read, Write};

const CRC32_U8_TABLE: [u32; 256] = [
	0x0000_0000,0x04C1_1DB7,0x0982_3B6E,0x0D43_26D9,0x1304_76DC,0x17C5_6B6B,0x1A86_4DB2,0x1E47_5005,
	0x2608_EDB8,0x22C9_F00F,0x2F8A_D6D6,0x2B4B_CB61,0x350C_9B64,0x31CD_86D3,0x3C8E_A00A,0x384F_BDBD,
	0x4C11_DB70,0x48D0_C6C7,0x4593_E01E,0x4152_FDA9,0x5F15_ADAC,0x5BD4_B01B,0x5697_96C2,0x5256_8B75,
	0x6A19_36C8,0x6ED8_2B7F,0x639B_0DA6,0x675A_1011,0x791D_4014,0x7DDC_5DA3,0x709F_7B7A,0x745E_66CD,
	0x9823_B6E0,0x9CE2_AB57,0x91A1_8D8E,0x9560_9039,0x8B27_C03C,0x8FE6_DD8B,0x82A5_FB52,0x8664_E6E5,
	0xBE2B_5B58,0xBAEA_46EF,0xB7A9_6036,0xB368_7D81,0xAD2F_2D84,0xA9EE_3033,0xA4AD_16EA,0xA06C_0B5D,
	0xD432_6D90,0xD0F3_7027,0xDDB0_56FE,0xD971_4B49,0xC736_1B4C,0xC3F7_06FB,0xCEB4_2022,0xCA75_3D95,
	0xF23A_8028,0xF6FB_9D9F,0xFBB8_BB46,0xFF79_A6F1,0xE13E_F6F4,0xE5FF_EB43,0xE8BC_CD9A,0xEC7D_D02D,
	0x3486_7077,0x3047_6DC0,0x3D04_4B19,0x39C5_56AE,0x2782_06AB,0x2343_1B1C,0x2E00_3DC5,0x2AC1_2072,
	0x128E_9DCF,0x164F_8078,0x1B0C_A6A1,0x1FCD_BB16,0x018A_EB13,0x054B_F6A4,0x0808_D07D,0x0CC9_CDCA,
	0x7897_AB07,0x7C56_B6B0,0x7115_9069,0x75D4_8DDE,0x6B93_DDDB,0x6F52_C06C,0x6211_E6B5,0x66D0_FB02,
	0x5E9F_46BF,0x5A5E_5B08,0x571D_7DD1,0x53DC_6066,0x4D9B_3063,0x495A_2DD4,0x4419_0B0D,0x40D8_16BA,
	0xACA5_C697,0xA864_DB20,0xA527_FDF9,0xA1E6_E04E,0xBFA1_B04B,0xBB60_ADFC,0xB623_8B25,0xB2E2_9692,
	0x8AAD_2B2F,0x8E6C_3698,0x832F_1041,0x87EE_0DF6,0x99A9_5DF3,0x9D68_4044,0x902B_669D,0x94EA_7B2A,
	0xE0B4_1DE7,0xE475_0050,0xE936_2689,0xEDF7_3B3E,0xF3B0_6B3B,0xF771_768C,0xFA32_5055,0xFEF3_4DE2,
	0xC6BC_F05F,0xC27D_EDE8,0xCF3E_CB31,0xCBFF_D686,0xD5B8_8683,0xD179_9B34,0xDC3A_BDED,0xD8FB_A05A,
	0x690C_E0EE,0x6DCD_FD59,0x608E_DB80,0x644F_C637,0x7A08_9632,0x7EC9_8B85,0x738A_AD5C,0x774B_B0EB,
	0x4F04_0D56,0x4BC5_10E1,0x4686_3638,0x4247_2B8F,0x5C00_7B8A,0x58C1_663D,0x5582_40E4,0x5143_5D53,
	0x251D_3B9E,0x21DC_2629,0x2C9F_00F0,0x285E_1D47,0x3619_4D42,0x32D8_50F5,0x3F9B_762C,0x3B5A_6B9B,
	0x0315_D626,0x07D4_CB91,0x0A97_ED48,0x0E56_F0FF,0x1011_A0FA,0x14D0_BD4D,0x1993_9B94,0x1D52_8623,
	0xF12F_560E,0xF5EE_4BB9,0xF8AD_6D60,0xFC6C_70D7,0xE22B_20D2,0xE6EA_3D65,0xEBA9_1BBC,0xEF68_060B,
	0xD727_BBB6,0xD3E6_A601,0xDEA5_80D8,0xDA64_9D6F,0xC423_CD6A,0xC0E2_D0DD,0xCDA1_F604,0xC960_EBB3,
	0xBD3E_8D7E,0xB9FF_90C9,0xB4BC_B610,0xB07D_ABA7,0xAE3A_FBA2,0xAAFB_E615,0xA7B8_C0CC,0xA379_DD7B,
	0x9B36_60C6,0x9FF7_7D71,0x92B4_5BA8,0x9675_461F,0x8832_161A,0x8CF3_0BAD,0x81B0_2D74,0x8571_30C3,
	0x5D8A_9099,0x594B_8D2E,0x5408_ABF7,0x50C9_B640,0x4E8E_E645,0x4A4F_FBF2,0x470C_DD2B,0x43CD_C09C,
	0x7B82_7D21,0x7F43_6096,0x7200_464F,0x76C1_5BF8,0x6886_0BFD,0x6C47_164A,0x6104_3093,0x65C5_2D24,
	0x119B_4BE9,0x155A_565E,0x1819_7087,0x1CD8_6D30,0x029F_3D35,0x065E_2082,0x0B1D_065B,0x0FDC_1BEC,
	0x3793_A651,0x3352_BBE6,0x3E11_9D3F,0x3AD0_8088,0x2497_D08D,0x2056_CD3A,0x2D15_EBE3,0x29D4_F654,
	0xC5A9_2679,0xC168_3BCE,0xCC2B_1D17,0xC8EA_00A0,0xD6AD_50A5,0xD26C_4D12,0xDF2F_6BCB,0xDBEE_767C,
	0xE3A1_CBC1,0xE760_D676,0xEA23_F0AF,0xEEE2_ED18,0xF0A5_BD1D,0xF464_A0AA,0xF927_8673,0xFDE6_9BC4,
	0x89B8_FD09,0x8D79_E0BE,0x803A_C667,0x84FB_DBD0,0x9ABC_8BD5,0x9E7D_9662,0x933E_B0BB,0x97FF_AD0C,
	0xAFB0_10B1,0xAB71_0D06,0xA632_2BDF,0xA2F3_3668,0xBCB4_666D,0xB875_7BDA,0xB536_5D03,0xB1F7_40B4
];

/// CRC-32 using lookup table
///
fn crc32(init: u32, data: &[u8]) -> u32 {
	let mut rem: u32 = init;
	for b in data {
		let idx = ((rem >> 24) ^ (*b as u32)) & 0xFF;

		rem = (rem << 8) ^ CRC32_U8_TABLE[idx as usize];
	}

	rem
}

fn create_uf2(offset: u32, data: &[u8]) -> Vec<u8> {
	const PAYLOAD_SIZE: u32 = 256;
	const PAYLOAD_BYTES: [u8; 4] = PAYLOAD_SIZE.to_le_bytes();
	const MAGIC1: [u8; 4] = [0x55, 0x46, 0x32, 0x0A];
	const MAGIC2: [u8; 4] = [0x57, 0x51, 0x5D, 0x9E];
	const MAGIC3: [u8; 4] = [0x30, 0x6F, 0xB1, 0x0A];
	const FLAGS: [u8; 4] = [0x00, 0x20, 0x00, 0x00]; // familyID present flag
	const FAMILYID: [u8; 4] = [0x56, 0xFF, 0x8B, 0xE4]; // Pi pico family
	const PADDING: [u8; 220] = [0; 220];

	let num_blocks = (data.len() as u32) / PAYLOAD_SIZE;
	let nblocks: [u8; 4] = num_blocks.to_le_bytes();

	// Scratch vector per loop
	let block = &mut Vec::new();
	block.reserve_exact(512);

	// Output vector
	let mut out = Vec::new();
	out.reserve_exact((num_blocks * 512) as usize);

	for i in 0..num_blocks {
		// Offset of this particular block (assuming all data is contiguous)
		let block_offset = offset + i * PAYLOAD_SIZE;

		// Add header info
		// TODO: Move as much out as possible, and use copy_from_slice inside loop
		block.extend_from_slice(&MAGIC1);
		block.extend_from_slice(&MAGIC2);
		block.extend_from_slice(&FLAGS);
		block.extend_from_slice(&block_offset.to_le_bytes());
		block.extend_from_slice(&PAYLOAD_BYTES);
		block.extend_from_slice(&(i as u32).to_le_bytes());
		block.extend_from_slice(&nblocks);
		block.extend_from_slice(&FAMILYID);
		assert_eq!(block.len(), 32);

		// Add data
		block.extend_from_slice(&data[(256*i as usize) .. (256*(i+1) as usize)]);

		// Pad and add final magic
		block.extend_from_slice(&PADDING);
		block.extend_from_slice(&MAGIC3);

		// End of loop stuff
		assert_eq!(block.len(), 512);
		out.extend_from_slice(&block);
		block.clear();
		println!("Created block {}", i+1);
	}

	out
}

fn main() {
	let mut data: [u8; 256] = [0;256];

	// Get input file
	let argv = env::args().collect::<Vec<String>>();
	if argv.len() != 2 {
		eprintln!("Usage: uf2 input_file");
		return;
	}

	// Open file and read data
	let fname = &argv[1];
	let f = &mut fs::File::open(fname).expect("Could not open data file");
	let _ = f.read(&mut data[..252]);

	// 2 bytes data (Infinite loop)
	//data[0] = 0xFE;
	//data[1] = 0xE7;

	// Checksum
	let check = crc32(0xFFFF_FFFF, &data[..252]);
	println!("Check is 0x{check:08X}");

	// Checksum (data + all zeros)
	data[252..].copy_from_slice(&check.to_le_bytes());
	let offset = 0x1000_0000;

	// Make UF2
	let out = create_uf2(offset, &data);

	// Save UF2 to file
	let f = &mut fs::File::create("/tmp/out.uf2").expect("Could not open file to write");
	f.write(&out).expect("Write failed");
}
