//use nix;
//use socket2;

use std::net::Ipv4Addr;


enum IpProto {
	Icmp,
	Igmp,
	Tcp,
	Udp,
	Experimental,
	Reserved,
}
impl From<isize> for IpProto {
	fn from(v: isize) -> Self {
		use IpProto::*;
		match v {
			1 => Icmp,
			2 => Igmp,
			6 => Tcp,
			17 => Udp,
			253 | 254 => Experimental,
			_ => Reserved,
		}
	}
}

struct IpOption {
	copied: bool,
	class: u8,
	num: u8,
	len: u8,
	data: Vec<u8>,
}

struct Ipv4Header {
	_version: u8, // Always 4
	_ihl: u8, // [5,15]
	dscp: u8,
	ecn: u8,
	length: u16, // [20, 65535]
	id: u16,
	flags: u8,
	fragment_offset: u16,
	ttl: u8,
	protocol: IpProto,
	checksum: u16,
	src: Ipv4Addr,
	dst: Ipv4Addr,
	options: Vec<IpOption>
}
struct Ipv4Packet {
	header: Ipv4Header,
	payload: Vec<u8>,
}
