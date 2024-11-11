#![allow(dead_code)]

pub const XIP_SSI_BASE: u32 = 0x1800_0000;
pub const XIP_SSI_CTRLR0: u32 = XIP_SSI_BASE;
pub const XIP_SSI_CTRLR1: u32 = XIP_SSI_BASE + 0x04;
pub const XIP_SSI_SSIENR: u32 = XIP_SSI_BASE + 0x08;
pub const XIP_SSI_BAUDR: u32 = XIP_SSI_BASE + 0x14;
pub const XIP_SSI_SPI_CTRLR0: u32 = XIP_SSI_BASE + 0xf4;

pub const RESETS_BASE: u32 = 0x4000_C000;
pub const RESETS_RESET: u32 = RESETS_BASE + 0x0;
pub const RESETS_WDSEL: u32 = RESETS_BASE + 0x4;
pub const RESETS_RESET_DONE: u32 = RESETS_BASE + 0x8;

pub const PSM_BASE: u32 = 0x4001_0000;
pub const PSM_FRCE_ON: u32 = PSM_BASE + 0x0;
pub const PSM_FRCE_OFF: u32 = PSM_BASE + 0x4;
pub const PSM_WDSEL: u32 = PSM_BASE + 0x8;
pub const PSM_DONE: u32 = PSM_BASE + 0xC;

pub const IO_BANK0_BASE: u32 = 0x4001_4000;
pub const GPIO0_STATUS: u32 = IO_BANK0_BASE + 0x000;
pub const GPIO0_CTRL: u32 = IO_BANK0_BASE + 0x004;

pub const GPIO25_STATUS: u32 = IO_BANK0_BASE + 0x0C8; // LED pin
pub const GPIO25_CTRL: u32 = IO_BANK0_BASE + 0x0CC;

pub const PADS_BANK0_BASE: u32 = 0x4001_C000;
pub const PADS_BANK0_GPIO25: u32 = PADS_BANK0_BASE + 0x68;

pub const GPIO_F5: u32 = 0x5;
pub const GPIO_FUNC_SIO: u32 = GPIO_F5;
pub const GPIO_FUNC_NULL: u32 = 0x1F;
