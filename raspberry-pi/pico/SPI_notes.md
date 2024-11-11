# Notes on SPI to try to get it working :/
## Pico general
Standard flash chip is Winbond W25Q16JV, per pico datasheet.


## 1.4 Pinout Reference (pg 11)
Pins 51-56 are QSPI.
* 51 - QSPI_SD3
* 52 - QSPI_SCLK
* 53 - QSPI_SD0
* 54 - QSPI_SD2
* 55 - QSPI_SD1
* 56 - QSPI_SS_N

## 2.1 Bus Fabric (pg 15)
All data paths are 32 bits wide.

Flash XIP has both a direct AHB connection, and a shared connection through the
 AHB splitter. Split is shared with PIO0, PIO1, USB, and DMA.

SPI0 and SPI1 accessed through APB Bridge and APB splitter.

The four bus masters (Core 0, Core 1, DMA controller read and write ports) can
 access any four *different* crossbar ports simultaneously.

### 2.1.1.1 Bus Priority
Bus priority dictates behavior of bus arbiters, controlled by `BUS_PRIORITY` in
 `BUSCTRL` register block.

### 2.1.1.2 Bus Performance Counters
Bus performance counters exist for profiling/debugging.

### 2.1.2 Atomic register access
Register blocks allocated 4kB of address space, with decode methods as follows:

* `Addr + 0x0000` - Normal read write access
* `Addr + 0x1000` - Atomic XOR on write
* `Addr + 0x2000` - Atomic bitmask set on write
* `Addr + 0x3000` - Atomic bitmask clear on write

SPI uses bus interposer to do "atomic" operations, which extends access time by
 two clock cycles.

### 2.1.3 APB Bridge
APB access have cycle penalties as follows:
* APB bus accesses take two cycles minimum (setup phase and access phase)
* The bridge adds an additional cycle to read accesses, as the bus request and
   response are registered
* The bridge adds **two** additional cycles to write accesses, as the APB setup
   phase cannot begin until the AHB-Lite write data is valid.

## 2.2 Address Map
### 2.2.1 Summary
| Name | Addr |
|---|---|
| ROM | `0x00000000` |
| XIP | `0x10000000` |
| SRAM | `0x20000000` |
| APB Peripherals | `0x40000000` |
| AHB-Lite Peripherals | `0x50000000` |
| IOPORT Registers | `0xD0000000` |
| Cortex-M0+ internal registers | `0xE0000000` |

### 2.2.2 Detail
Subset of detail relevant for this

| Name | Addr |
|---|---|
| `XIP_CTRL_BASE` | `0x14000000` |
| `XIP_SSI_BASE` | `0x18000000` |
| `SYSINFO_BASE` | `0x40000000` |
| `SYSCFG_BASE` | `0x40004000` |
| `RESETS_BASE` | `0x4000C000` |
| `PSM_BASE` | `0x40010000` |
| `IO_BANK0_BASE` | `0x40014000` |
| `IO_QSPI_BASE` | `0x40018000` |
| `PADS_BANK0_BASE` | `0x4001C000` |
| `PADS_QSPI_BASE` | `0x40020000` |
| `SPI0_BASE` | `0x4003C000` |
| `SPI1_BASE` | `0x40040000` |
| `XIP_AUX_BASE` | `0x50400000` |

## 2.6 Memory
### 2.6.1 ROM
Contains:
* Initial startup routine
* Flash boot sequence
* Flash programming routines
* USB mass storage device with UF2 support
* Utility libraries such as fast floating point

### 2.6.3.3 SSI
XIP provided by SSI interface supports 1, 2, or 4-bit SPI flash interfaces and
 can insert either an instruction prefix or mode continuation bits on each XIP
 access. Standard (but slowest) is issue 03h (hex) serial flash read command
 for each access. Max SPI clock frequency is half system clock. Default
 clockdiv setting is 4.


## 2.8 Bootrom
### 2.8.1.3 Flash second stage
The flash second stage must configure the SSI and external flash for best
 possible execute-in-place (XIP) performance. This includes interface width,
 SCK frequency, SPI instruction prefix and an XIP continuation code for
 address-data only modes.

Alternatively, the second stage can simply shadow an image from external flash
 into SRAM and avoid configuring XIP.

## 4.10.10.6 XIP mode support in SPI mode
XIP is enabled in `DW_apb_ssi` when the XIP cache is enabled. This control
 signal indicates whether APB transfers are register read-write or XIP reads.
 TODO: Determine whether cache needs to be enabled for register configuration.

**XIP operation is only supported in enhanced SPI modes (Dual, Quad)** of
 operation, therefore `CTRLR0.SPI_FRF` should not be 0.

For an XIP read operation:

1. Set the SPI frame format and data frame size value in `CTRLR0` register.
    Note that the value of the maximum data frame size is 32.
2. Set the address length, wait cycles, and a transaction type in the
    `SPI_CTRLR0` register. Note that the maximum address length is 32.

After these settings, a user can initiate a read transaction through the APB
 interface which will be transfered to the SPI peripheral using the programmed
 values. `N` is 1, 3, and 7 for SPI mode, Dual, and Quad modes, respectively.
