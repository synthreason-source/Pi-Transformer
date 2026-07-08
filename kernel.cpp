/*
 * OPTIMIZED KERNEL WITH GRAPHICS STATE MANAGEMENT
 * ================================================
 * All graphics rendering uses atomic frame composition
 * Unified color palette prevents inconsistencies
 * State machine ensures complete frames with no trailing
 */
 
#include <cstddef>
#include <cstdarg>
#include <cstdint>
// =============================================================================
// SECTION 1: TYPE DEFS, STDLIB/CXX STUBS, AND LOW-LEVEL FUNCTIONS
// =============================================================================
#define SECTOR_SIZE 512
// Process structure for ELF execution
#define MAX_ELF_PROCESSES 4
#define ELF_STACK_SIZE (64 * 1024)  // 64KB stack per process
#define ELF_HEAP_SIZE (256 * 1024)   // 256KB heap per process
// --- Type Definitions ---
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef unsigned int uintptr_t;
typedef unsigned int size_t;
typedef signed long long int64_t;
// --- CXX ABI Stubs ---
namespace __cxxabiv1 {
    extern "C" int __cxa_guard_acquire(long long *g) { return !*(char *)(g); }
    extern "C" void __cxa_guard_release(long long *g) { *(char *)g = 1;}
    extern "C" void __cxa_pure_virtual() {}
    extern "C" void __cxa_throw_bad_array_new_length() {
        asm volatile("cli; hlt");
    }
    class __class_type_info { virtual void dummy(); };
    void __class_type_info::dummy() {}
    class __si_class_type_info { virtual void dummy(); };
    void __si_class_type_info::dummy() {}
}
extern "C" {
    void* memcpy(void* dest, const void* src, size_t n) { 
        uint8_t* d = (uint8_t*)dest; 
        const uint8_t* s = (const uint8_t*)src; 
        for (size_t i = 0; i < n; i++) d[i] = s[i]; 
        return dest; 
    }

    void* memset(void* ptr, int value, size_t num) { 
        uint8_t* p = (uint8_t*)ptr; 
        for (size_t i = 0; i < num; i++) p[i] = (uint8_t)value; 
        return ptr; 
    }

    void* memmove(void* dest, const void* src, size_t n) {
        uint8_t* d = (uint8_t*)dest;
        const uint8_t* s = (const uint8_t*)src;
        if (d < s) {
            for (size_t i = 0; i < n; i++) d[i] = s[i];
        } else {
            for (size_t i = n; i != 0; i--) d[i-1] = s[i-1];
        }
        return dest;
    }
}

extern "C" unsigned long long __udivmoddi4(unsigned long long num,
                                           unsigned long long den,
                                           unsigned long long *rem)
{
    if (den == 0) {
        if (rem) *rem = 0;
        return 0;
    }

    unsigned long long q = 0;
    unsigned long long r = 0;

    for (int i = 63; i >= 0; --i) {
        r <<= 1;
        r |= (num >> i) & 1ULL;
        if (r >= den) {
            r -= den;
            q |= (1ULL << i);
        }
    }

    if (rem) *rem = r;
    return q;
}

extern "C" long long __divmoddi4(long long num,
                                 long long den,
                                 long long *rem)
{
    if (den == 0) {
        if (rem) *rem = 0;
        return 0;
    }

    bool neg_q = (num < 0) ^ (den < 0);
    bool neg_r = (num < 0);

    unsigned long long unum = (num < 0) ? (unsigned long long)(-num) : (unsigned long long)num;
    unsigned long long uden = (den < 0) ? (unsigned long long)(-den) : (unsigned long long)den;

    unsigned long long ur = 0;
    unsigned long long uq = __udivmoddi4(unum, uden, &ur);

    long long q = neg_q ? -(long long)uq : (long long)uq;
    long long r = neg_r ? -(long long)ur : (long long)ur;

    if (rem) *rem = r;
    return q;
}


// --- Forward Declarations ---
class Window;
class TerminalWindow;
class FileExplorerWindow; // New
extern "C" void kernel_main(uint32_t magic, uint32_t multiboot_addr);
void launch_new_terminal();
void launch_new_explorer(); // New
void launch_terminal_with_command(const char* command); // ADD THIS LINE

int fat32_write_file(const char* filename, const void* data, uint32_t size);
int fat32_remove_file(const char* filename);
char* fat32_read_file_as_string(const char* filename);
void fat32_list_files();
typedef struct { char name[11]; uint8_t attr; uint8_t ntres; uint8_t crt_time_tenth; uint16_t crt_time, crt_date, lst_acc_date, fst_clus_hi; uint16_t wrt_time, wrt_date, fst_clus_lo; uint32_t file_size; } __attribute__((packed)) fat_dir_entry_t;
int fat32_list_directory(const char* path, fat_dir_entry_t* buffer, int max_entries);
int fat32_find_entry(const char* filename, fat_dir_entry_t* entry_out, uint32_t* sector_out, uint32_t* offset_out);
int fat32_stat_file(const char* filename, uint32_t* size_out);
bool fat32_init();


// objcopy --rename-section emits these as address labels into .rodata.
// Must be declared as incomplete arrays (extern "C" uint8_t name[]) so
// that the identifiers decay to pointers without needing &.
// Using scalar uint8_t and then taking & also resolves, but treating a
// linker label as a single-byte object is undefined behaviour in C++.
extern "C" uint8_t ramdisk_start[];
extern "C" uint8_t ramdisk_end[];
extern "C" uint8_t hello_start[];
extern "C" uint8_t hello_end[];

bool extract_busybox_to_filesystem() {
    uint8_t* start = ramdisk_start;
    uint8_t* end   = ramdisk_end;
    if (end <= start) return false;
    uint32_t size  = (uint32_t)(end - start);
    // Sanity: must be a plausible ELF (>= 52 bytes), not absurdly large.
    if (size < 52 || size > 8 * 1024 * 1024) {
        return false;
    }
    // Check ELF magic before writing to avoid storing garbage on disk.
    if (start[0] != 0x7f || start[1] != 'E' ||
        start[2] != 'L'  || start[3] != 'F') {
        return false;
    }
    // If a "busybox" file already exists with the right size, skip the write
    // to save time on repeated boots.
    fat_dir_entry_t existing;
    uint32_t esec = 0, eoff = 0;
    if (fat32_find_entry("busybox", &existing, &esec, &eoff) == 0) {
        if (existing.file_size == size) return true;  // already current
    }
    int result = fat32_write_file("busybox", start, size);
    return (result == 0);
}

// Write the embedded hello test ELF to FAT32 as "hello".
// Same shape as extract_busybox_to_filesystem.
bool extract_hello_to_filesystem() {
    uint8_t* start = hello_start;
    uint8_t* end   = hello_end;
    if (end <= start) return false;
    uint32_t size  = (uint32_t)(end - start);
    if (size < 52 || size > 1 * 1024 * 1024) return false;
    if (start[0] != 0x7f || start[1] != 'E' ||
        start[2] != 'L'  || start[3] != 'F') return false;

    fat_dir_entry_t existing;
    uint32_t esec = 0, eoff = 0;
    if (fat32_find_entry("hello", &existing, &esec, &eoff) == 0) {
        if (existing.file_size == size) return true;
    }
    int result = fat32_write_file("hello", start, size);
    return (result == 0);
}
// --- Global Clipboard ---
static char g_clipboard_buffer[1024] = {0}; // New

// --- Low-level I/O functions ---
static inline void outb(uint16_t port, uint8_t val) { asm volatile ("outb %0, %1" : : "a"(val), "d"(port)); }
static inline void outl(uint16_t port, uint32_t val) { asm volatile ("outl %0, %1" : : "a"(val), "d"(port)); }
static inline uint8_t inb(uint16_t port) { uint8_t ret; asm volatile ("inb %1, %0" : "=a"(ret) : "d"(port)); return ret; }
static inline uint32_t inl(uint16_t port) { uint32_t ret; asm volatile ("inl %1, %0" : "=a"(ret) : "d"(port)); return ret; }
static inline uint32_t pci_read_config_dword(uint16_t bus, uint8_t device, uint8_t function, uint8_t offset) {
    uint32_t address = 0x80000000 | ((uint32_t)bus << 16) | ((uint32_t)device << 11) | ((uint32_t)function << 8) | (offset & 0xFC);
    outl(0xCF8, address);
    return inl(0xCFC);
}
// =============================================================================
//  CENTRAL DEFINITIONS & FORWARD DECLARATIONS
// =============================================================================
	
// =============================================================================
// INDEPENDENT RUN AND EXEC IMPLEMENTATION
// =============================================================================
// This refactoring completely decouples run and exec processes:
// - run: manages disk-based object files with full disk I/O context
// - exec: manages in-memory compiled code with no disk dependencies
// - Separate process tables, separate resource management
// - No shared state between the two subsystems

// =============================================================================
// SECTION 1: SEPARATE PROCESS CONTEXTS
// =============================================================================


// COMPLETE FIXED KERNEL.CPP - BUSYBOX ELF EXEC READY
// Paste this ENTIRE file as your new kernel.cpp. Compiles clean.
// All warnings/errors fixed. Busybox takes terminal control.
// No compiler. Pure ELF loader + x86 emu + ring buffers.

// ===== INCLUDES & DEFS =====
#include <cstddef>
#include <cstdarg>
#include <cstdint>

// Your existing includes/types/stdlib from paste.txt...
// SECTORSIZE removed - use SECTOR_SIZE (defined above)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
// =============================================================================
// CORRECT ORDER - paste these blocks in this sequence
// =============================================================================

// --- STEP 1: Constants - move these to the TOP, before any class ---
#define INBUFSIZE   512
#define OUTBUFSIZE  4096
#define SB          0x80000000u
#define MAXelf_processes 4
#define ELFSTACKSIZE     (64  * 1024)
#define ELFHEAPSIZE      (256 * 1024)

// --- STEP 2: ElfProcess struct - move before TerminalWindow ---
struct ElfProcess {
    int input_pos = 0;

    uint32_t entry_point = 0;   // full virtual address (e_entry)
    uint32_t vaddr_base  = 0;   // min PT_LOAD vaddr (== physical base in Bochs)
    uint32_t vaddr_end   = 0;   // max PT_LOAD vaddr + memsz (exclusive)
    uint8_t* memory_base = nullptr;
    uint32_t memory_size = 0;
    uint8_t* stack       = nullptr;
    uint32_t esp = 0, eip = 0;
    TerminalWindow* terminal = nullptr;
    char cmdline[256] = {0};
    bool waiting_for_input = false;
    bool completed = false;
    int exit_code  = 0;
    bool active = false;
    bool cpu_initialized = false;
    
    unsigned int brk_addr = 0;
    char inbuf[INBUFSIZE];   int in_head=0,  in_tail=0;
    char outbuf[OUTBUFSIZE]; int out_head=0, out_tail=0;
};

// --- STEP 3: Global array - move before TerminalWindow ---

static ElfProcess elf_processes[MAX_ELF_PROCESSES];

// --- STEP 4: Ring buffer helpers - move before TerminalWindow ---
bool in_empty(int slot) {
    return elf_processes[slot].in_head == elf_processes[slot].in_tail;
}
bool out_empty(int slot) {
    return elf_processes[slot].out_head == elf_processes[slot].out_tail;
}
void push_input(int slot, char c) {
    ElfProcess& p = elf_processes[slot];
    int next = p.in_head + 1;
    if (next == INBUFSIZE) next = 0;
    if (next != p.in_tail) {
        p.inbuf[p.in_head] = c;
        p.in_head = next;
    }
}
char pop_input(int slot) {
    ElfProcess& p = elf_processes[slot];
    if (in_empty(slot)) return 0;
    char c = p.inbuf[p.in_tail];
    p.in_tail = (p.in_tail + 1) % INBUFSIZE;
    return c;
}
void push_output(int slot, char c) {
    ElfProcess& p = elf_processes[slot];
    int next = p.out_head + 1;
    if (next == OUTBUFSIZE) next = 0;
    if (next != p.out_tail) {
        p.outbuf[p.out_head] = c;
        p.out_head = next;
    }
}
char pop_output(int slot) {
    ElfProcess& p = elf_processes[slot];
    if (out_empty(slot)) return 0;
    char c = p.outbuf[p.out_tail];
    p.out_tail = (p.out_tail + 1) % OUTBUFSIZE;
    return c;
}

// --- STEP 5: Bochs externs - move before TerminalWindow ---
extern "C" void bochs_set_process_memory(
    uint8_t* base, uint32_t size, uint32_t vaddr_base);
extern "C" void bochs_cpu_init();
extern "C" void bochs_cpu_prewarm();
extern "C" void bochs_cpu_set_eip(uint32_t eip);
extern "C" void bochs_cpu_set_esp(uint32_t esp);
extern "C" int  bochs_cpu_tick(int steps);
extern "C" uint32_t bochs_cpu_get_eax();
extern "C" uint32_t bochs_cpu_get_eip();
// New: slot management, brk, I/O callbacks, and input-wait detection
extern "C" void bochs_activate_slot(int slot);
extern "C" void bochs_finalize_process_memory();
extern "C" void bochs_set_brk(int slot, uint32_t brk_addr);
extern "C" void bochs_register_io_callbacks(
    int slot,
    int  (*read_cb )(int),
    void (*write_cb)(int, char),
    void (*exit_cb )(int, int));
extern "C" bool bochs_process_wants_input(int slot);

// ── In-kernel TCC compiler (tcc_kernel.cpp + i386-libtcc-kern.a) ─────────────
// tcc_kernel_version() == 0 → stub (no TCC), 2 → real in-kernel TCC.
extern "C" int  tcc_kernel_version(void);
extern "C" void tcc_kernel_cmd_cc(void* terminal, const char* src_name,
                                  const char* out_name);
// Heavy "reset everything" — called between ELF runs so each launch
// starts from the same state as the first one. See the comment block
// over its definition in bochs_glue.cpp for the full rationale.
extern "C" void bochs_reset_all_slots();

// Surgical per-slot release — called when ONE slot's process exits
// but other slots are still running. Wipes only that slot's glue
// state (mapping, mem_base pointer, saved CPU snapshot). Unlike
// bochs_reset_all_slots(), does NOT touch BX_CPU(0), so peer slots
// keep executing safely. Must be called BEFORE the kernel frees the
// slot's backing slab so the mapping_unregister still has a valid
// vaddr range to look up.
extern "C" void bochs_release_slot(int slot);

// Forward declarations for the ELF loader helpers and IO callbacks defined
// later in this file. busybox/hello use the same lazy-init pattern as
// load_and_execute_elf (cpu_initialized=false, no bochs_cpu_init here).
// start_elf_process/load_elf_image_to_slab are kept for test_main usage.
static bool load_elf_image_to_slab(int slot, const unsigned char* elf,
                                   unsigned int elf_size, unsigned int& entry_out);
static bool start_elf_process(int slot, const unsigned char* elf,
                              unsigned int elf_size);
// IO callbacks — forward-declared so the `test` command handler can
// restore them after test_module_run() overwrites slot 0's callbacks.
static int  elf_io_read (int slot);
static void elf_io_write(int slot, char c);
static void elf_io_exit (int slot, int code);

// test_module.cpp holds the test execution code (formerly the standalone
// test_main.cpp). It is activated by typing `test` in a terminal. The module
// renders a three-row "VGA-style" overlay (breadcrumbs / fault tag / GUEST
// row) by calling back through a TestSink — it has no knowledge of windows.
//
// The kernel side, below, owns:
//   - g_test_vga[]   : a 3x80 cell buffer the module writes via vga_cell()
//   - test_sink_*    : the C callbacks the module invokes
//   - g_test_overlay_owner : which TerminalWindow currently shows the overlay
// The TerminalWindow::draw() method paints g_test_vga[] as three colored
// rows at the top of its content area whenever it owns the overlay.
#include "test_module.h"

// =====================================================================
// C++ global constructor (__init_array) walk — runs ONCE at boot.
//
// boot.S jumps straight from BSS-zero into kernel_main; it does NOT walk
// __init_array. That means the file-scope C++ constructors that build
// the Bochs core objects (bx_cpu(0), bx_mem, the CPUID parameter
// objects in bochs_infra.cpp, icache's pageWriteStampTable, ...) never
// run unless something explicitly walks the array.
//
// Previously the ONLY caller was test_module_run() (via its private
// run_init_array_once()). So a freshly booted system that launched an
// ELF in the Bochs emulator window *without first typing `test`* called
// bochs_cpu_init() -> BX_CPU(0)->initialize() against raw, zero-filled
// BSS objects — null vtables — and instantly faulted. The emulator
// window's guest trapped out through the port-0xE8 exit stub on its
// very first tick, so the window appeared to "crash / autoclose" the
// first time it was used.
//
// The fix: walk __init_array here, once, from kernel_main, before any
// Bochs entry point can be reached. test_module's own guard
// (g_init_array_done) plus our call to test_module_mark_ctors_done()
// guarantees the constructors are never run a second time by `test`.
extern "C" void (*__init_array_start[])();
extern "C" void (*__init_array_end[])();

static bool g_kernel_ctors_done = false;

static void kernel_run_global_ctors_once() {
    if (g_kernel_ctors_done) return;
    g_kernel_ctors_done = true;
    for (void (**p)() = __init_array_start; p < __init_array_end; ++p) {
        if (*p) (*p)();
    }
}

struct TestVgaCell { char ch; uint8_t attr; };
static TestVgaCell g_test_vga[3][80];
static bool        g_test_overlay_active = false;
// Forward decl: set to the TerminalWindow that ran `test`. Declared void*
// here because TerminalWindow is defined much further down; the command
// handler casts it back.
static void*       g_test_overlay_owner  = nullptr;

// Clear the overlay buffer to blank grey-on-black cells.
static void test_vga_clear() {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 80; ++c) {
            g_test_vga[r][c].ch   = ' ';
            g_test_vga[r][c].attr = 0x0F;
        }
}

// --- TestSink callbacks (C linkage so the module can take their address) ---
// vga_cell: the module's faithful reproduction of writing VGA text memory.
extern "C" void test_sink_vga_cell(int row, int col, char ch, uint8_t attr) {
    if (row < 0 || row >= 3 || col < 0 || col >= 80) return;
    g_test_vga[row][col].ch   = ch;
    g_test_vga[row][col].attr = attr;
    g_test_overlay_active     = true;
}
// put_line: forwarded to the owning terminal's console_print (defined after
// TerminalWindow, since it needs the full class — see test_sink_put_line()).
extern "C" void test_sink_put_line(const char* s);
// flush: repaints the whole screen and swaps buffers mid-test. Defined
// after WindowManager / swap_buffers are available (see test_sink_flush()).
// Without this the GUI would freeze for the whole blocking test run and
// the overlay would never become visible.
extern "C" void test_sink_flush(void);

// --- STEP 6: NOW TerminalWindow and everything else follows ---
// Moved here to be visible to all classes and functions
static bool    g_fs_encryption_enabled = false;
static uint8_t g_fs_xor_key[64]        = {0};  // 64-byte keystream
static int     g_fs_xor_key_len        = 0;

// =============================================================================
// FILESYSTEM XOR ENCRYPTION LAYER
// =============================================================================

// Derive a 64-byte keystream from password using FNV-1a mixing
static void derive_xor_key(const char* password, uint8_t key_out[64], int* key_len_out) {
    uint32_t h = 2166136261u;
    const char* p = password;
    while (*p) {
        h ^= (uint8_t)*p++;
        h *= 16777619u;
    }

    // Expand hash into 64 bytes by re-mixing with position
    for (int i = 0; i < 64; i++) {
        h ^= (uint32_t)i * 2654435761u;
        h *= 16777619u;
        key_out[i] = (uint8_t)(h ^ (h >> 16));
    }
    *key_len_out = 64;
}

// XOR a 512-byte sector buffer in place.
// LBA is mixed into each block so identical plaintext sectors at different
// locations produce different ciphertext (poor-man's sector tweak).
static void xor_sector(uint8_t* buf, uint64_t lba) {
    if (!g_fs_encryption_enabled || g_fs_xor_key_len == 0) return;

    // Build a per-sector tweak from the LBA
    uint8_t tweak[8];
    for (int i = 0; i < 8; i++) tweak[i] = (uint8_t)(lba >> (i * 8));

    for (int i = 0; i < SECTOR_SIZE; i++) {
        uint8_t k = g_fs_xor_key[i % g_fs_xor_key_len];
        uint8_t t = tweak[i % 8];
        buf[i] ^= (k ^ t);
    }
}

// Call after successful unlock to arm the encryption layer
static void fs_crypto_init(const char* password) {
    derive_xor_key(password, g_fs_xor_key, &g_fs_xor_key_len);
    g_fs_encryption_enabled = true;
}

// Call on lock/disk-switch to wipe key material from memory
static void fs_crypto_clear() {
    g_fs_encryption_enabled = false;
    memset(g_fs_xor_key, 0, sizeof(g_fs_xor_key));
    g_fs_xor_key_len = 0;
}
// --- Global State Variables ---
// Moved here to be visible to all classes
static volatile uint32_t g_timer_ticks = 0;

// --- Forward Declarations ---
class Window;
class TerminalWindow;
class FileExplorerWindow;

// Kernel Entry
extern "C" void kernel_main(uint32_t magic, uint32_t multiboot_addr);

// App Launchers
void launch_new_terminal();
void launch_new_explorer();

// FAT32 Function Prototypes
int fat32_write_file(const char* filename, const void* data, uint32_t size);
int fat32_remove_file(const char* filename);
char* fat32_read_file_as_string(const char* filename);
void fat32_list_files();
bool fat32_init();
void fat32_get_fne_from_entry(fat_dir_entry_t* entry, char* out); // New helper
int fat32_stat_file(const char* filename, uint32_t* size_out); // Guest disk wrapper helper (bochs_glue.cpp)
// --- Minimal Standard Library ---
size_t strlen(const char* str) { size_t len = 0; while (str[len]) len++; return len; }
int memcmp(const void* ptr1, const void* ptr2, size_t n) { const uint8_t* p1 = (const uint8_t*)ptr1; const uint8_t* p2 = (const uint8_t*)ptr2; for(size_t i=0; i<n; ++i) if(p1[i] != p2[i]) return p1[i] - p2[i]; return 0; }
int strcmp(const char* s1, const char* s2) { while(*s1 && (*s1 == *s2)) { s1++; s2++; } return *(const unsigned char*)s1 - *(const unsigned char*)s2; }
int strncmp(const char* s1, const char* s2, size_t n) { if (n == 0) return 0; do { if (*s1 != *s2++) return *(unsigned const char*)s1 - *(unsigned const char*)--s2; if (*s1++ == 0) break; } while (--n != 0); return 0; }
char* strchr(const char* s, int c) { while (*s != (char)c) if (!*s++) return nullptr; return (char*)s; }
char* strrchr(const char* s, int c) { const char* last = nullptr; do { if (*s == (char)c) last = s; } while (*s++); return (char*)last; } // New for finding extensions
char* strcpy(char *dest, const char *src) { char *ret = dest; while ((*dest++ = *src++)); return ret; }
char* strncpy(char* dest, const char* src, size_t n) { size_t i; for (i = 0; i < n && src[i] != '\0'; i++) dest[i] = src[i]; for ( ; i < n; i++) dest[i] = '\0'; return dest; }
char* strcat(char* dest, const char* src) {
    char* ptr = dest;
    while (*ptr != '\0') { ptr++; }
    while (*src != '\0') { *ptr = *src; ptr++; src++; }
    *ptr = '\0';
    return dest;
}
char* strncat(char *dest, const char *src, size_t n) {
    size_t dest_len = strlen(dest);
    size_t i;
    for (i = 0 ; i < n && src[i] != '\0' ; i++)
        dest[dest_len + i] = src[i];
    dest[dest_len + i] = '\0';
    return dest;
}
int simple_atoi(const char* str) { int res = 0; while(*str >= '0' && *str <= '9') { res = res * 10 + (*str - '0'); str++; } return res; }
const char* strstr(const char* haystack, const char* needle) {
    if (!*needle) return haystack;
    const char* p1 = haystack;
    while (*p1) {
        const char* p1_begin = p1;
        const char* p2 = needle;
        while (*p1 && *p2 && *p1 == *p2) { p1++; p2++; }
        if (!*p2) { return p1_begin; }
        p1 = p1_begin + 1;
    }
    return nullptr;
}
int snprintf(char* buffer, size_t size, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char* buf = buffer;
    char* end = buffer + size - 1;
    while (*fmt && buf < end) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd') {
                int val = va_arg(args, int);
                char tmp[32];
                char* t = tmp + 31; *t = '\0';
                bool neg = val < 0;
                if(neg) val = -val;
                if(val == 0) *--t = '0';
                else while(val > 0) { *--t = '0' + val % 10; val /= 10; }
                if (neg) *--t = '-';
                while (*t && buf < end) *buf++ = *t++;
            } else if (*fmt == 's') {
                const char* s = va_arg(args, const char*);
                while (*s && buf < end) *buf++ = *s++;
            } else if (*fmt == 'c') {
                char c = (char)va_arg(args, int);
                if (buf < end) *buf++ = c;
            } else {
                 if (buf < end) *buf++ = *fmt;
            }
        } else {
            *buf++ = *fmt;
        }
        fmt++;
    }
    *buf = '\0';
    va_end(args);
    return buf - buffer;
}

// --- Basic Memory Allocator ---
// Single 32MB global heap in BSS (not stack!) — enough for BusyBox + 4 ELF procs + FAT32 + backbuffer
/* 16 MB heap — keeps total BSS under ~20 MB so GRUB can zero it reliably.
   100 MB caused bochs init_memory to corrupt/loop because GRUB only zeroes
   ~12–24 MB of BSS before jumping to _start; FreeListAllocator nodes beyond
   that boundary contained garbage.  Budget: BusyBox ramdisk (~2 MB mapped
   read-only), 4 ELF slabs × ~300 KB, backbuffer 3 MB, FAT32 sector
   buffers, Bochs init_memory internals (~2 MB).  16 MB covers all of that
   with room to spare and keeps total BSS well within GRUB's zeroing window. */
static uint8_t kernel_heap[64 * 1024 * 1024];
static size_t heap_ptr = 0;
void* operator new(size_t, void* p) { return p; }

// Pulled forward from below so oom_halt() (which lives in this file before
// the original include site) can paint glyphs to the live framebuffer.
#include "font.h"

// Forward decl of the framebuffer descriptor so oom_halt() can paint to the
// live framebuffer below. Real definition (with initializer) is further down.
struct FramebufferInfo { uint32_t* ptr; uint32_t width, height, pitch; };
extern FramebufferInfo fb_info;

class FreeListAllocator {
public:
    struct FreeBlock {
        size_t size;
        FreeBlock* next;
    };

private:
    FreeBlock* freeListHead;

public:
    FreeListAllocator() : freeListHead(nullptr) {}

    void init(void* heapStart, size_t heapSize) {
        if (!heapStart || heapSize < sizeof(FreeBlock)) {
            return;
        }
        freeListHead = static_cast<FreeBlock*>(heapStart);
        freeListHead->size = heapSize;
        freeListHead->next = nullptr;
    }

    void* allocate(size_t size) {
        size_t required_size = (size + sizeof(size_t) + (alignof(FreeBlock) - 1)) & ~(alignof(FreeBlock) - 1);
        if (required_size < sizeof(FreeBlock)) {
            required_size = sizeof(FreeBlock);
        }

        FreeBlock* prev = nullptr;
        FreeBlock* current = freeListHead;
        while (current) {
            if (current->size >= required_size) {
                if (current->size >= required_size + sizeof(FreeBlock)) {
                    FreeBlock* newBlock = (FreeBlock*)((char*)current + required_size);
                    newBlock->size = current->size - required_size;
                    newBlock->next = current->next;

                    if (prev) {
                        prev->next = newBlock;
                    } else {
                        freeListHead = newBlock;
                    }
                } else {
                    required_size = current->size;
                    if (prev) {
                        prev->next = current->next;
                    } else {
                        freeListHead = current->next;
                    }
                }
                
                *(size_t*)current = required_size;
                return (char*)current + sizeof(size_t);
            }
            prev = current;
            current = current->next;
        }
        return nullptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;

        FreeBlock* block_to_free = (FreeBlock*)((char*)ptr - sizeof(size_t));
        size_t block_size = *(size_t*)block_to_free;
        block_to_free->size = block_size;

        FreeBlock* prev = nullptr;
        FreeBlock* current = freeListHead;
        while (current && current < block_to_free) {
            prev = current;
            current = current->next;
        }

        if (prev) {
            prev->next = block_to_free;
        } else {
            freeListHead = block_to_free;
        }
        block_to_free->next = current;

        if (block_to_free->next && (char*)block_to_free + block_to_free->size == (char*)block_to_free->next) {
            block_to_free->size += block_to_free->next->size;
            block_to_free->next = block_to_free->next->next;
        }

        if (prev && (char*)prev + prev->size == (char*)block_to_free) {
            prev->size += block_to_free->size;
            prev->next = block_to_free->next;
        }
    }

    // Sum of all free blocks currently on the free list. Used by callers
    // (e.g. chkdsk) that are about to request a large allocation and want
    // to fail gracefully instead of triggering oom_halt(), which freezes
    // the whole kernel rather than just the requesting operation.
    size_t total_free() const {
        size_t sum = 0;
        for (FreeBlock* cur = freeListHead; cur; cur = cur->next) sum += cur->size;
        return sum;
    }
};

static FreeListAllocator g_allocator;

// Write OOM message directly to VGA text buffer AND to the live framebuffer.
// VGA text alone is invisible in graphics mode unless draw_vga_overlay() runs
// — but oom_halt() is reached from inside an allocation site that's about to
// halt the kernel, so the main loop never paints again. Painting straight to
// fb_info.ptr (live, NOT backbuffer) bypasses the swap_buffers cycle so the
// message is visible immediately even with a dead main loop.
static void oom_halt(size_t size) {
    // ── 1. VGA text plane (forensic, visible only if overlay paints later) ──
    volatile char* vga = (volatile char*)0xB8000;
    const char* msg = "OOM HALT";
    for (int i = 0; msg[i]; i++) { vga[i*2] = msg[i]; vga[i*2+1] = 0x4F; }
    // Print size in decimal after the message
    char buf[16]; int n = 0, s = (int)size;
    if (s == 0) buf[n++] = '0';
    else { int tmp = s; int d = 1; while (tmp >= 10) { tmp /= 10; d++; }
           for (int i = d-1; i >= 0; i--) { buf[i] = '0' + s%10; s/=10; n++; } }
    buf[n] = 0;
    int off = 8;
    vga[off*2]=' '; vga[off*2+1]=0x4F; off++;
    for (int i = 0; i < n; i++) { vga[(off+i)*2]=buf[i]; vga[(off+i)*2+1]=0x4F; }

    // ── 2. Live framebuffer (visible immediately, survives a hung main loop) ──
    if (fb_info.ptr) {
        // Bright red bar across rows 24..47 — distinct from host_fault_handler's
        // bar (rows 0..23) so we can tell OOM apart from CPU faults at a glance.
        int bar_y0 = 24;
        int bar_h  = 24;
        if (bar_y0 + bar_h > (int)fb_info.height) bar_h = fb_info.height - bar_y0;
        if (bar_h > 0) {
            for (int y = bar_y0; y < bar_y0 + bar_h; ++y) {
                uint32_t* row = &fb_info.ptr[y * (fb_info.pitch / 4)];
                for (uint32_t x = 0; x < fb_info.width; ++x) row[x] = 0xC00000u;
            }
        }
        auto put_glyph = [](char ch, int x0, int y0, uint32_t color) {
            if ((unsigned char)ch > 127) return;
            if (x0 + 8 > (int)fb_info.width)  return;
            if (y0 + 8 > (int)fb_info.height) return;
            const uint8_t* glyph = font + (int)ch * 8;
            for (int yy = 0; yy < 8; ++yy) {
                uint32_t* row = &fb_info.ptr[(y0 + yy) * (fb_info.pitch / 4) + x0];
                uint8_t bits = glyph[yy];
                for (int xx = 0; xx < 8; ++xx) {
                    row[xx] = (bits & (0x80 >> xx)) ? color : 0xC00000u;
                }
            }
        };
        const char* prefix = "OOM HALT ";
        int x = 8;
        int y = bar_y0 + 8;
        for (int i = 0; prefix[i]; ++i) { put_glyph(prefix[i], x, y, 0xFFFFFFu); x += 8; }
        for (int i = 0; i < n;       ++i) { put_glyph(buf[i],    x, y, 0xFFFFFFu); x += 8; }
    }

    asm volatile("cli");
    for(;;) asm volatile("hlt");
}

// Bump-pool fallback, implemented in bochs_cstubs.c. The Bochs ctors do
// large allocations (icache.o's pageWriteStampTable ctor needs 4 MiB)
// through the global operator new; when the kernel's FreeListAllocator
// is exhausted we fall back to the dedicated 48 MiB Bochs pool instead
// of halting. bochs_pool_owns() lets operator delete recognise a pointer
// that came from that pool (the bump allocator does not free per-object).
extern "C" void* bochs_pool_alloc(size_t n);
extern "C" int   bochs_pool_owns(const void* p);

void* operator new(size_t size) {
    void* p = g_allocator.allocate(size);
    if (!p) p = bochs_pool_alloc(size);   // fall back to the Bochs pool
    if (!p) oom_halt(size);               // both exhausted — now halt
    return p;
}

void* operator new[](size_t size) {
    return operator new(size);
}

void operator delete(void* ptr) noexcept {
    if (!ptr) return;
    if (bochs_pool_owns(ptr)) return;     // bump-pool memory: never freed
    g_allocator.deallocate(ptr);
}

void operator delete[](void* ptr) noexcept {
    operator delete(ptr);
}

void operator delete(void* ptr, size_t size) noexcept {
    (void)size;
    operator delete(ptr);
}

void operator delete[](void* ptr, size_t size) noexcept {
    (void)size;
    operator delete[](ptr);
}


// ── Non-halting allocation for callers that must be able to fail cleanly ──
// kernel_alloc_nofail()/kernel_free() are the same FreeListAllocator +
// Bochs-pool fallback as operator new/delete above, EXCEPT they return
// nullptr on exhaustion instead of calling oom_halt(). Anything driven by
// untrusted/arbitrary user input — most notably the in-kernel TCC compiler
// in tcc_kernel.cpp, where a malformed or oversized source file can cause
// runaway allocation — must use this instead of plain `new`, otherwise a
// single bad `cc <file.c>` permanently freezes the entire OS rather than
// just failing that one command.
extern "C" void* kernel_alloc_nofail(size_t size) {
    void* p = g_allocator.allocate(size);
    if (!p) p = bochs_pool_alloc(size);
    return p;   // may be nullptr — caller must check
}

extern "C" void kernel_free(void* ptr) {
    if (!ptr) return;
    if (bochs_pool_owns(ptr)) return;
    g_allocator.deallocate(ptr);
}

// Returns the number of bytes actually usable at `ptr` (i.e. safe to read
// or write), or 0 if unknown (e.g. pointer came from the Bochs bump pool,
// which keeps no per-allocation size). The FreeListAllocator stores the
// rounded-up block size (including its own header) immediately before the
// pointer it returned, so we can recover a safe upper bound here without
// any extra bookkeeping. Used by tcc_kernel.cpp's realloc() so growing a
// buffer never reads past the end of the smaller, original allocation —
// that out-of-bounds read previously corrupted adjacent heap memory on
// every realloc-to-grow call.
extern "C" size_t kernel_alloc_usable_size(void* ptr) {
    if (!ptr) return 0;
    if (bochs_pool_owns(ptr)) return 0;   // bump pool: no recoverable size
    size_t block_size = *(size_t*)((char*)ptr - sizeof(size_t));
    if (block_size <= sizeof(size_t)) return 0;  // corrupt/garbage guard
    return block_size - sizeof(size_t);
}

// ── Non-halting byte-buffer alloc/free for ELF process images ─────────────
// load_and_execute_elf() previously allocated the guest image slab and its
// stack with plain `new uint8_t[...]`, which goes through operator new and
// therefore HALTS THE ENTIRE KERNEL on exhaustion (oom_halt()) — exactly
// like the in-kernel TCC compiler did before it was switched to
// kernel_alloc_nofail. A user launching an ELF when the heap happens to be
// tight (e.g. several Bochs slots already running, or a large compiled
// program) should see "cc: out of memory" / a failed launch, not freeze
// the whole OS over one `cc foo.c && foo`.
//
// These wrap kernel_alloc_nofail/kernel_free in a uint8_t* interface so
// load_and_execute_elf can swap its `new[]`/`delete[]` calls 1:1 without
// touching the unrelated `new[]`/`delete[]` call sites used for filesystem
// I/O buffers elsewhere in this file (those are kernel-internal and not
// driven by arbitrary user input in the same way).
extern "C" uint8_t* elf_alloc_bytes(size_t size) {
    return (uint8_t*)kernel_alloc_nofail(size);
}
extern "C" void elf_free_bytes(uint8_t* ptr) {
    kernel_free(ptr);
}



// =============================================================================
// SECTION 2: BOOTLOADER INFO, FONT, RTC
// =============================================================================
struct multiboot_info {
    uint32_t flags, mem_lower, mem_upper, boot_device, cmdline, mods_count, mods_addr;
    uint32_t syms[4], mmap_length, mmap_addr;
    uint32_t drives_length, drives_addr, config_table, boot_loader_name, apm_table;
    uint32_t vbe_control_info, vbe_mode_info;
    uint16_t vbe_mode, vbe_interface_seg, vbe_interface_off, vbe_interface_len;
    uint64_t framebuffer_addr;
    uint32_t framebuffer_pitch, framebuffer_width, framebuffer_height;
    uint8_t framebuffer_bpp, framebuffer_type, color_info[6];
} __attribute__((packed));

uint8_t rtc_read(uint8_t reg) { outb(0x70, reg); return inb(0x71); }
uint8_t bcd_to_bin(uint8_t val) { return ((val / 16) * 10) + (val & 0x0F); }
struct RTC_Time { uint8_t second, minute, hour, day, month; uint16_t year; };
RTC_Time read_rtc() {
    RTC_Time t;
    uint8_t century = 20;
    while (rtc_read(0x0A) & 0x80);
    uint8_t regB = rtc_read(0x0B);
    bool is_bcd = !(regB & 0x04);
    t.second = rtc_read(0x00); t.minute = rtc_read(0x02); t.hour = rtc_read(0x04);
    t.day = rtc_read(0x07); t.month = rtc_read(0x08); t.year = rtc_read(0x09);
    if (is_bcd) {
        t.second = bcd_to_bin(t.second); t.minute = bcd_to_bin(t.minute); t.hour = bcd_to_bin(t.hour);
        t.day = bcd_to_bin(t.day); t.month = bcd_to_bin(t.month); t.year = bcd_to_bin(t.year);
    }
    t.year += century * 100;
    return t;
}

// =============================================================================
// SECTION 3: GRAPHICS & WINDOWING SYSTEM WITH STATE MANAGEMENT
// =============================================================================

/* Back-buffer: 1024x768x4 = 3 MB.  Declared as a static array in BSS so it
   doesn't consume heap space.  fb_info dimensions are checked before use. */
static uint32_t backbuffer_storage[1024 * 768];
static uint32_t* backbuffer = backbuffer_storage;
// FramebufferInfo struct is forward-declared near the top of this file (above
// oom_halt). Here we just define the single instance.
FramebufferInfo fb_info;

// =============================================================================
// UNIFIED COLOR PALETTE - PREVENTS COLOR INCONSISTENCIES
// =============================================================================
namespace ColorPalette {
    // Desktop colors
    constexpr uint32_t DESKTOP_TEAL      = 0x008080;
    constexpr uint32_t DESKTOP_BLUE      = 0x00004B; 
    constexpr uint32_t DESKTOP_GRAY      = 0x404040;
    
    // Taskbar colors
    constexpr uint32_t TASKBAR_GRAY      = 0x808080;
    constexpr uint32_t TASKBAR_DARK      = 0x606060;
    constexpr uint32_t TASKBAR_LIGHT     = 0xC0C0C0;
    
    // Window colors
    constexpr uint32_t WINDOW_BG         = 0x000000;
    constexpr uint32_t WINDOW_BORDER     = 0xC0C0C0;
    constexpr uint32_t TITLEBAR_ACTIVE   = 0x000080;
    constexpr uint32_t TITLEBAR_INACTIVE = 0x808080;
    constexpr uint32_t FILE_EXPLORER_BG  = 0xFFFFFF; // New
    
    // Button colors
    constexpr uint32_t BUTTON_FACE       = 0xC0C0C0;
    constexpr uint32_t BUTTON_HIGHLIGHT  = 0xFFFFFF;
    constexpr uint32_t BUTTON_SHADOW     = 0x808080;
    constexpr uint32_t BUTTON_CLOSE      = 0xFF0000;
    
    // Text colors
    constexpr uint32_t TEXT_BLACK        = 0x000000;
    constexpr uint32_t TEXT_WHITE        = 0xFFFFFF;
    constexpr uint32_t TEXT_GREEN        = 0x00FF00;
    constexpr uint32_t TEXT_GRAY         = 0x808080;
    
    // Cursor color
    constexpr uint32_t CURSOR_WHITE      = 0xFFFFFF;

    // Icon Colors
    constexpr uint32_t ICON_FILE_FILL    = 0xFFF1B5; // Light yellow
    constexpr uint32_t ICON_FILE_OUTLINE = 0x808080;
    constexpr uint32_t ICON_FOLDER_FILL  = 0xFFD3A1; // Light orange
    constexpr uint32_t ICON_SHORTCUT_ARROW = 0x0000FF; // Blue
}

// =============================================================================
// ENHANCED RENDER STATE MACHINE - ELIMINATES TRAILING AND ENSURES CONTINUITY
// =============================================================================

struct RenderState {
    // Frame state tracking
    uint32_t frameNumber;
    bool frameComplete;
    bool backgroundCleared;
    
    // Window rendering state
    int currentWindow;
    int renderPhase;
    
    // Progressive rendering within window
    int currentLine;
    int currentChar;
    int currentScanline;
    
    // Dirty tracking
    bool needsFullRedraw;
    bool windowsDirty;
    
    // Timing
    uint32_t lastFrameTick;
    uint32_t lastInputTick;
};

struct InputState {
    int byteIndex;
    uint8_t pendingBytes[16];
    int pendingCount;
    bool hasNewInput;
};

static RenderState g_render_state = {0, false, false, 0, 0, 0, 0, 0, true, true, 0, 0};
static InputState g_input_state = {0, {0}, 0, false};

// =============================================================================
// ENHANCED GRAPHICS DRIVER
// =============================================================================

inline int gfx_abs(int x) { return x < 0 ? -x : x; }

struct Color {
    uint8_t r, g, b, a;

    uint32_t to_rgb() const {
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    uint32_t to_bgr() const {
        return (a << 24) | (b << 16) | (g << 8) | r;
    }
};

namespace Colors {
    constexpr Color Black = {0, 0, 0, 255};
    constexpr Color White = {255, 255, 255, 255};
    constexpr Color Red = {255, 0, 0, 255};
    constexpr Color Green = {0, 255, 0, 255};
    constexpr Color Blue = {0, 0, 255, 255};
}

class GraphicsDriver;

class GraphicsDriver {
private:
    bool is_bgr_format;

    inline uint32_t convert_color(const Color& color) const {
        return is_bgr_format ? color.to_bgr() : color.to_rgb();
    }

    // This function converts a standard 0xRRGGBB color into 0xBBGGRR for BGR framebuffers
    inline uint32_t rgb_to_bgr(uint32_t color) const {
        if (!is_bgr_format) return color;

        uint8_t a = (color >> 24) & 0xFF;
        uint8_t r = (color >> 16) & 0xFF;
        uint8_t g = (color >> 8)  & 0xFF;
        uint8_t b = (color >> 0)  & 0xFF;

        return (a << 24) | (b << 16) | (g << 8) | r;
    }

public:
    GraphicsDriver() : is_bgr_format(true) {}

    void init(bool bgr_format = true) {
        is_bgr_format = bgr_format;
    }

    void clear_screen(uint32_t rgb_color) {
        if (!backbuffer || !fb_info.ptr) return;

        uint32_t color = rgb_to_bgr(rgb_color);
        uint32_t pixel_count = fb_info.width * fb_info.height;

        #ifdef __i386__
        uint32_t* target = backbuffer;
        asm volatile(
            "rep stosl"
            : "=D"(target), "=c"(pixel_count)
            : "D"(target), "c"(pixel_count), "a"(color)
            : "memory"
        );
        #else
        for (uint32_t i = 0; i < pixel_count; i++) {
            backbuffer[i] = color;
        }
        #endif
    }

    void clear_screen(const Color& color) {
        clear_screen(convert_color(color));
    }

    void put_pixel(int x, int y, uint32_t rgb_color) {
        if (backbuffer && x >= 0 && x < (int)fb_info.width && y >= 0 && y < (int)fb_info.height) {
            backbuffer[y * fb_info.width + x] = rgb_to_bgr(rgb_color);
        }
    }

    void put_pixel(int x, int y, const Color& color) {
        put_pixel(x, y, convert_color(color));
    }

    void draw_line(int x0, int y0, int x1, int y1, const Color& color) {
        int dx = gfx_abs(x1 - x0);
        int dy = gfx_abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            put_pixel(x0, y0, color);

            if (x0 == x1 && y0 == y1) break;

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    void draw_rect(int x, int y, int w, int h, const Color& color) {
        for (int i = 0; i < w; i++) {
            put_pixel(x + i, y, color);
            put_pixel(x + i, y + h - 1, color);
        }
        for (int i = 0; i < h; i++) {
            put_pixel(x, y + i, color);
            put_pixel(x + w - 1, y + i, color);
        }
    }

    void fill_rect(int x, int y, int w, int h, const Color& color) {
        uint32_t col = convert_color(color);
        for (int dy = 0; dy < h; dy++) {
            for (int dx = 0; dx < w; dx++) {
                put_pixel(x + dx, y + dy, col);
            }
        }
    }
};

static GraphicsDriver g_gfx;

void put_pixel_back(int x, int y, uint32_t color) {
    if (backbuffer && x >= 0 && x < (int)fb_info.width && y >= 0 && y < (int)fb_info.height) {
        backbuffer[y * fb_info.width + x] = color;
    }
}

void draw_char(char c, int x, int y, uint32_t color) {
    if ((unsigned char)c > 127) return;
    const uint8_t* glyph = font + (int)c * 8;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if ((glyph[i] & (0x80 >> j))) {
                put_pixel_back(x + j, y + i, color);
            }
        }
    }
}

void draw_string(const char* str, int x, int y, uint32_t color) {
    for (int i = 0; str[i]; i++) {
        draw_char(str[i], x + i * 8, y, color);
    }
}

// ─── Host CPU fault handler (called from boot.S isr_common) ────────────
//
// boot.S installs a 256-entry IDT pointing at stubs that all chain to
// isr_common. isr_common writes a VGA-text-mode breadcrumb at row 1 ('!'
// + hex vector) then calls THIS function with the vector number, then
// halts forever. We render the same info onto the live framebuffer so
// the user can see the fault tag even after graphics mode hides VGA
// text. This is a one-shot, no-return diagnostic — we never resume from
// a host fault.
extern "C" volatile unsigned char bx_panic_breadcrumbs[64];

// Pull faulting EIP from the on-stack exception frame. The IDT stub in
// boot.S pushes the standard CPU error frame (err, eip, cs, eflags, ...);
// we read eip via a tiny inline-asm helper that walks back up from the
// current frame. This is best-effort — if frame layout changes, eip just
// reads as 0 and we still get the breadcrumb trail, which is the more
// useful signal anyway.
static inline unsigned read_caller_eip(void) {
    unsigned eip = 0;
    __asm__ volatile(
        "movl 4(%%ebp), %0\n"   // return addr into host_fault_handler == stub
        : "=r"(eip));
    return eip;
}

extern "C" void host_fault_handler(unsigned vector) {
    // Read the faulting EIP up front so we can hand it to the test
    // module too — a #UD/GP inside libcpu.a is only diagnosable if we
    // know WHERE it faulted, not just the vector.
    unsigned caller_eip = read_caller_eip();

    // If a `test` self-test is running, mirror the fault into its
    // window overlay (row 1, "!XX  EIP=...") before the kernel's own
    // rendering, and emit the vector+EIP to the host debug console.
    if (test_module_active()) test_module_fault((int)vector, caller_eip);

    if (!fb_info.ptr) return;

    auto put_glyph = [](char ch, int x0, int y0, uint32_t color, uint32_t bg) {
        if ((unsigned char)ch > 127) ch = '?';
        if (x0 + 8 > (int)fb_info.width)  return;
        if (y0 + 8 > (int)fb_info.height) return;
        const uint8_t* glyph = font + (int)ch * 8;
        for (int yy = 0; yy < 8; ++yy) {
            uint32_t* row = &fb_info.ptr[(y0 + yy) * (fb_info.pitch / 4) + x0];
            uint8_t bits = glyph[yy];
            for (int xx = 0; xx < 8; ++xx) {
                row[xx] = (bits & (0x80 >> xx)) ? color : bg;
            }
        }
    };

    auto hex = [](unsigned n) -> char {
        return (char)((n < 10) ? ('0' + n) : ('A' + (n - 10)));
    };

    // Bright red bar across the top — impossible to miss. Two rows now,
    // so we have room for the EIP + breadcrumb trail.
    int bar_h = 40;
    if (bar_h > (int)fb_info.height) bar_h = (int)fb_info.height;
    for (int y = 0; y < bar_h; ++y) {
        uint32_t* row = &fb_info.ptr[y * (fb_info.pitch / 4)];
        for (uint32_t x = 0; x < fb_info.width; ++x) row[x] = 0xC00000u;
    }

    // Row 1: "HOST FAULT !XX  EIP=XXXXXXXX"
    {
        const char* msg = "HOST FAULT !";
        int x = 8, y = 4;
        for (int i = 0; msg[i]; ++i) {
            put_glyph(msg[i], x, y, 0xFFFFFFu, 0xC00000u); x += 8;
        }
        put_glyph(hex((vector >> 4) & 0xF), x, y, 0xFFFFFFu, 0xC00000u); x += 8;
        put_glyph(hex( vector       & 0xF), x, y, 0xFFFFFFu, 0xC00000u); x += 16;

        const char* eipmsg = "EIP=";
        for (int i = 0; eipmsg[i]; ++i) {
            put_glyph(eipmsg[i], x, y, 0xFFFFFFu, 0xC00000u); x += 8;
        }
        for (int i = 7; i >= 0; --i) {
            put_glyph(hex((caller_eip >> (i * 4)) & 0xF), x, y, 0xFFFFFFu, 0xC00000u);
            x += 8;
        }
    }

    // Row 2: dump bx_panic_breadcrumbs trail (Bochs init progress markers).
    // Last printable char = last successful step inside the Bochs glue
    // before the fault. Empty/zero means fault happened before Bochs init
    // started — the bug is in the kernel↔Bochs call path, not Bochs itself.
    {
        const char* lbl = "BX:";
        int x = 8, y = 20;
        for (int i = 0; lbl[i]; ++i) {
            put_glyph(lbl[i], x, y, 0xFFFFFFu, 0xC00000u); x += 8;
        }
        for (int i = 0; i < 48; ++i) {
            unsigned char c = bx_panic_breadcrumbs[i];
            put_glyph(c ? (char)c : '.', x, y, 0xFFFF00u, 0xC00000u);
            x += 8;
        }
    }

    // Row 2 (right side): mirror x86_tick's VGA breadcrumb row so the user
    // sees both trails on one screen. VGA text at 0xB8000 row 2 cols 0..15.
    {
        int x = 8 + 8 * 4 + 8 * 48 + 16;
        int y = 20;
        const char* lbl = "TK:";
        for (int i = 0; lbl[i]; ++i) {
            put_glyph(lbl[i], x, y, 0xFFFFFFu, 0xC00000u); x += 8;
        }
        volatile unsigned short* vga = (volatile unsigned short*)(0xB8000 + 2 * 80);
        for (int i = 0; i < 16 && x + 8 <= (int)fb_info.width; ++i) {
            unsigned char c = (unsigned char)(vga[i] & 0xFF);
            put_glyph(c ? (char)c : '.', x, y, 0x00FF00u, 0xC00000u);
            x += 8;
        }
    }
}

// ─── Live framebuffer breadcrumb ───────────────────────────────────────────
// Paints a single 8x8 glyph DIRECTLY to the live framebuffer (skipping the
// backbuffer / swap_buffers path) at row 0, col `slot` (each slot 8 pixels
// wide). Designed to be visible even when the kernel main loop hangs, since
// it bypasses the per-frame paint cycle entirely.
//
// Used to localise freezes inside Bochs glue: each interesting step calls
// live_breadcrumb with a different slot+char, so the LAST char visible
// before the hang identifies the last successful step.
//
// Layout convention: slots 0..15 are reserved for bochs_glue diagnostics;
// rendered at fb x=col*8, y=0 with a black background tile.
extern "C" void live_breadcrumb(int slot, char ch) {
    // If a `test` self-test is running, mirror each breadcrumb into its
    // window overlay (row 0, white-on-blue) as well as the framebuffer.
    if (test_module_active()) test_module_breadcrumb(slot, ch);

    if (!fb_info.ptr) return;
    if (slot < 0 || slot >= 80) return;

    int x0 = slot * 8;
    int y0 = 0;
    if (x0 + 8 > (int)fb_info.width)  return;
    if (y0 + 8 > (int)fb_info.height) return;

    if ((unsigned char)ch > 127) ch = '?';
    const uint8_t* glyph = font + (int)ch * 8;

    for (int yy = 0; yy < 8; ++yy) {
        uint32_t* row = &fb_info.ptr[(y0 + yy) * (fb_info.pitch / 4) + x0];
        uint8_t bits = glyph[yy];
        for (int xx = 0; xx < 8; ++xx) {
            row[xx] = (bits & (0x80 >> xx)) ? 0xFFFF00u   /* yellow on */
                                            : 0x000080u;  /* dark blue bg */
        }
    }
}

// ─── Diagnostic overlay: mirror VGA text mode (rows 0 / 1 / 2) onto the
// framebuffer ────────────────────────────────────────────────────────────────
//
// The kernel writes diagnostic breadcrumbs to VGA text memory at 0xB8000:
//   row 0: boot trace ('B','S','Z','C'), heartbeat at col 79, panic tag
//          at col 70 (from bx_recover), Bochs tick markers at col 72/73.
//   row 1: host-IDT fault tag '!XX' (from boot.S isr_common).
//   row 2: x86_tick lazy-init progress (L,M,I,S,E,B,T,t).
//
// Once the framebuffer is initialised these writes are invisible because
// graphics mode hides the VGA text plane. This overlay reads the first
// 80 cells of rows 0/1/2 every frame and draws them as a 24-pixel strip
// across the top of the framebuffer, so any breadcrumb that gets written
// is visible immediately.
extern "C" void draw_vga_overlay() {
    if (!backbuffer || !fb_info.ptr) return;

    // Black backdrop bar (inline to avoid forward-decl on draw_rect_filled).
    {
        int bar_h = 24;     // 3 rows of 8px
        if (bar_h > (int)fb_info.height) bar_h = (int)fb_info.height;
        for (int y = 0; y < bar_h; ++y) {
            uint32_t* row = &backbuffer[y * fb_info.width];
            for (uint32_t x = 0; x < fb_info.width; ++x) row[x] = 0x000000u;
        }
    }

    volatile const uint16_t* vga = (volatile const uint16_t*)0xB8000;

    auto vga_attr_to_rgb = [](uint8_t attr) -> uint32_t {
        uint8_t fg = attr & 0x0F;
        static const uint32_t fg_rgb[16] = {
            0x000000, 0x0000AA, 0x00AA00, 0x00AAAA,
            0xAA0000, 0xAA00AA, 0xAA5500, 0xAAAAAA,
            0x555555, 0x5555FF, 0x55FF55, 0x55FFFF,
            0xFF5555, 0xFF55FF, 0xFFFF55, 0xFFFFFF
        };
        return fg_rgb[fg];
    };

    // Detect emphasised backgrounds (0x4F = white-on-red, used for host
    // IDT and Bochs panic): promote those to bright red.
    auto cell_color = [&](uint16_t cell) -> uint32_t {
        uint8_t attr = (uint8_t)(cell >> 8);
        if ((attr & 0xF0) == 0x40) return 0xFF4040u;
        return vga_attr_to_rgb(attr);
    };

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 80; ++col) {
            uint16_t cell = vga[row * 80 + col];
            char ch = (char)(cell & 0xFF);
            if (ch == 0) continue;
            int x = col * 8;
            int y = row * 8;
            if (x + 8 > (int)fb_info.width)  break;
            if (y + 8 > (int)fb_info.height) break;
            draw_char(ch, x, y, cell_color(cell));
        }
    }
}

// =============================================================================
// OPTIMIZED FILL RECT - ATOMIC SCANLINE RENDERING
// =============================================================================
void draw_rect_filled(int x, int y, int w, int h, uint32_t color) {
    // Clip to screen bounds
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x >= (int)fb_info.width || y >= (int)fb_info.height) return;
    if (x + w > (int)fb_info.width) w = fb_info.width - x;
    if (y + h > (int)fb_info.height) h = fb_info.height - y;
    if (w <= 0 || h <= 0) return;

    // Render entire rect atomically (no state machine - prevents tearing)
    for (int dy = 0; dy < h; dy++) {
        int screenY = y + dy;
        if (screenY >= 0 && screenY < (int)fb_info.height) {
            uint32_t* row = &backbuffer[screenY * fb_info.width + x];
            
            // Fast fill with rep stosl on x86
            #ifdef __i386__
            uint32_t count = w;
            asm volatile(
                "rep stosl"
                : "=D"(row), "=c"(count)
                : "D"(row), "c"(count), "a"(color)
                : "memory"
            );
            #else
            for (int i = 0; i < w; i++) {
                row[i] = color;
            }
            #endif
        }
    }
}
#define FAT_ATTR_DIRECTORY 0x10
// =============================================================================
// PS/2 AND INPUT SYSTEM (Abbreviated - full implementation as before)
// =============================================================================

struct PS2State {
    uint32_t lastInputCheckTick;
    uint32_t lastOutputCheckTick;
    uint8_t inputAttemptCount;
    uint8_t outputAttemptCount;
};
static PS2State g_ps2state = {0, 0, 0, 0};

#define PS2_DATA_PORT       0x60
#define PS2_STATUS_PORT     0x64
#define PS2_COMMAND_PORT    0x64
#define PS2_CMD_READ_CONFIG     0x20
#define PS2_CMD_WRITE_CONFIG    0x60
#define PS2_CMD_DISABLE_PORT1   0xAD
#define PS2_CMD_ENABLE_PORT1    0xAE
#define PS2_CMD_DISABLE_PORT2   0xA7
#define PS2_CMD_ENABLE_PORT2    0xA8
#define PS2_CMD_TEST_PORT2      0xA9
#define PS2_CMD_TEST_CTRL       0xAA
#define PS2_CMD_WRITE_PORT2     0xD4
#define MOUSE_CMD_RESET         0xFF
#define MOUSE_CMD_RESEND        0xFE
#define MOUSE_CMD_SET_DEFAULTS  0xF6
#define MOUSE_CMD_DISABLE_DATA  0xF5
#define MOUSE_CMD_ENABLE_DATA   0xF4
#define MOUSE_CMD_SET_SAMPLE    0xF3
#define MOUSE_CMD_SET_RESOLUTION 0xE8
#define PS2_STATUS_OUTPUT_FULL  0x01
#define PS2_STATUS_INPUT_FULL   0x02
#define PS2_STATUS_AUX_DATA     0x20
#define PS2_STATUS_TIMEOUT      0x40
#define PS2_ACK                 0xFA
#define PS2_RESEND              0xFE

#define KEY_UP     -1
#define KEY_DOWN   -2
#define KEY_LEFT   -3
#define KEY_RIGHT  -4
#define KEY_DELETE -5
#define KEY_HOME   -6
#define KEY_END    -7

const char sc_ascii_nomod_map[]={0,0,'1','2','3','4','5','6','7','8','9','0','-','=','\b','\t','q','w','e','r','t','y','u','i','o','p','[',']','\n',0,'a','s','d','f','g','h','j','k','l',';','\'','`',0,'\\','z','x','c','v','b','n','m',',','.','/',0,0,0,' ',0};
const char sc_ascii_shift_map[]={0,0,'!','@','#','$','%','^','&','*','(',')','_','+','\b','\t','Q','W','E','R','T','Y','U','I','O','P','{','}','\n',0,'A','S','D','F','G','H','J','K','L',':','"','~',0,'|','Z','X','C','V','B','N','M','<','>','?',0,0,0,' ',0};
const char sc_ascii_ctrl_map[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,'\b','\t','\x11',0,0,0,0,0,0,0,0,'\x10',0,0,'\n',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,' ',0};

bool is_shift_pressed = false;
bool is_ctrl_pressed = false;
int mouse_x = 400, mouse_y = 300;
bool mouse_left_down = false;
bool mouse_left_last_frame = false;
bool mouse_right_down = false;       // New
bool mouse_right_last_frame = false; // New
char last_key_press = 0;

struct UniversalMouseState {
    int x;
    int y;
    bool left_button;
    bool right_button;
    bool middle_button;
    uint8_t packet_cycle;
    uint8_t packet_buffer[3];
    bool synchronized;
    bool initialized;
};

static UniversalMouseState universal_mouse_state = {400, 300, false, false, false, 0, {0}, false, false};

static void process_universal_mouse_packet(uint8_t data) {
    if (!universal_mouse_state.synchronized) {
        if (data & 0x08) {
            universal_mouse_state.packet_buffer[0] = data;
            universal_mouse_state.packet_cycle = 1;
            universal_mouse_state.synchronized = true;
            return;
        } else {
            return;
        }
    }
    
    universal_mouse_state.packet_buffer[universal_mouse_state.packet_cycle] = data;
    universal_mouse_state.packet_cycle++;
    
    if (universal_mouse_state.packet_cycle >= 3) {
        universal_mouse_state.packet_cycle = 0;
        
        uint8_t flags = universal_mouse_state.packet_buffer[0];
        
        if (!(flags & 0x08)) {
            universal_mouse_state.synchronized = false;
            return;
        }
        
        universal_mouse_state.left_button = flags & 0x01;
        universal_mouse_state.right_button = flags & 0x02;
        universal_mouse_state.middle_button = flags & 0x04;
        
        int8_t dx = (int8_t)universal_mouse_state.packet_buffer[1];
        int8_t dy = (int8_t)universal_mouse_state.packet_buffer[2];
        
        if (flags & 0x40) {
            dx = (dx > 0) ? 127 : -128;
        }
        if (flags & 0x80) {
            dy = (dy > 0) ? 127 : -128;
        }
        
        const int SENSITIVITY = 2;
        int move_x = dx * SENSITIVITY;
        int move_y = dy * SENSITIVITY;
        
        universal_mouse_state.x += move_x;
        universal_mouse_state.y -= move_y;
        
        if (universal_mouse_state.x < 0) universal_mouse_state.x = 0;
        if (universal_mouse_state.y < 0) universal_mouse_state.y = 0;
        if (universal_mouse_state.x >= (int)fb_info.width) 
            universal_mouse_state.x = fb_info.width - 1;
        if (universal_mouse_state.y >= (int)fb_info.height) 
            universal_mouse_state.y = fb_info.height - 1;
        
        universal_mouse_state.synchronized = true;
    }
}

// =============================================================================
// WINDOW SYSTEM
// =============================================================================

// New: Icon drawing functions
void draw_icon_file(int x, int y, bool is_shortcut) {
    draw_rect_filled(x, y, 32, 32, ColorPalette::ICON_FILE_FILL);
    draw_rect_filled(x, y, 32, 1, ColorPalette::ICON_FILE_OUTLINE);
    draw_rect_filled(x + 31, y, 1, 32, ColorPalette::ICON_FILE_OUTLINE);
    draw_rect_filled(x, y + 31, 32, 1, ColorPalette::ICON_FILE_OUTLINE);
    draw_rect_filled(x, y, 1, 32, ColorPalette::ICON_FILE_OUTLINE);
    if(is_shortcut) {
        draw_rect_filled(x + 4, y + 22, 10, 6, ColorPalette::ICON_SHORTCUT_ARROW);
        put_pixel_back(x+8, y+20, ColorPalette::ICON_SHORTCUT_ARROW);
        put_pixel_back(x+9, y+21, ColorPalette::ICON_SHORTCUT_ARROW);
    }
}

void draw_icon_folder(int x, int y) {
    draw_rect_filled(x, y + 5, 32, 27, ColorPalette::ICON_FOLDER_FILL);
    draw_rect_filled(x, y, 14, 8, ColorPalette::ICON_FOLDER_FILL);
    draw_rect_filled(x, y + 31, 32, 1, ColorPalette::ICON_FILE_OUTLINE);
}

// New: Desktop items structure
enum IconType { ICON_FILE, ICON_DIR, ICON_SHORTCUT, ICON_APP };
struct DesktopItem {
    char name[32];
    char path[128];
    int x, y;
    IconType type;
};

class Window {
public:
    int x, y, w, h;
    const char* title;
    bool has_focus;
    bool is_closed;

    Window(int x, int y, int w, int h, const char* title)
        : x(x), y(y), w(w), h(h), title(title), has_focus(false), is_closed(false) {}
    virtual ~Window() {}
    virtual void put_char(char c) {} // ADD THIS

    virtual void draw() = 0;
    virtual void on_key_press(char c) = 0;
    virtual void on_mouse_click(int mx, int my) {} // New
	virtual void on_mouse_right_click(int mx, int my) {} // ADD THIS LINE

    virtual void update() = 0;
    virtual void console_print(const char* s) {}
    virtual int  get_elf_slot() const { return -1; }  // overridden by TerminalWindow

    bool is_in_titlebar(int mx, int my) { return mx > x && mx < x + w && my > y && my < y + 25; }
    bool is_in_close_button(int mx, int my) { int btn_x = x + w - 22, btn_y = y + 4; return mx >= btn_x && mx < btn_x + 18 && my >= btn_y && my < btn_y + 18; }
    virtual void close() { is_closed = true; }
};

class WindowManager {
private:
    Window* windows[16];
    int num_windows;
    int focused_idx;
    int dragging_idx;
    int drag_offset_x, drag_offset_y;

    // New: Desktop & Context Menu management
    DesktopItem desktop_items[64];
    int num_desktop_items;
    int dragging_icon_idx;

    bool context_menu_active;
    int context_menu_x, context_menu_y;
	const char* context_menu_items[8];
    int num_context_menu_items;
    enum ContextType { CTX_DESKTOP, CTX_ICON, CTX_EXPLORER_ITEM }; // ADD CTX_EXPLORER_ITEM
    ContextType current_context;
    int context_icon_idx;
    char context_file_path[128]; // ADD THIS to store the file path    int num_context_menu_items;
    

public:
    WindowManager() : num_windows(0), focused_idx(-1), dragging_idx(-1), 
                      num_desktop_items(0), dragging_icon_idx(-1), 
                      context_menu_active(false) {}
    void show_file_context_menu(int mx, int my, const char* filename) {
		context_menu_active = true;
		context_menu_x = mx;
		context_menu_y = my;
		current_context = CTX_EXPLORER_ITEM;
		strncpy(context_file_path, filename, 127); // Store the filename for the action
		num_context_menu_items = 0;
		
		if (strstr(filename, ".obj") != nullptr || strstr(filename, ".OBJ") != nullptr) {
			context_menu_items[num_context_menu_items++] = "Run";
		}
		context_menu_items[num_context_menu_items++] = "Edit"; // ADDED THIS LINE
		context_menu_items[num_context_menu_items++] = "Create Shortcut";
		context_menu_items[num_context_menu_items++] = "Copy";
		context_menu_items[num_context_menu_items++] = "Delete";
	}
    // New: Load desktop items from filesystem
    // In WindowManager class
	 void print_to_window(int idx, const char* s) {
        if (idx >= 0 && idx < num_windows) {
            windows[idx]->console_print(s);
        }
    }
	 void put_char_to_focused(char c) {
        if (focused_idx >= 0 && focused_idx < num_windows) {
            windows[focused_idx]->put_char(c);
        }
    }
void load_desktop_items() {
    num_desktop_items = 0;

    // Load items from the root directory
    static fat_dir_entry_t file_list[64]; // Max 64 files on desktop
    int num_files = fat32_list_directory("/", file_list, 64);

    for (int i = 0; i < num_files && num_desktop_items < 64; ++i) {
        fat32_get_fne_from_entry(&file_list[i], desktop_items[num_desktop_items].name);
        strcpy(desktop_items[num_desktop_items].path, desktop_items[num_desktop_items].name);
        
        desktop_items[num_desktop_items].x = 30 + (num_desktop_items % 10) * 70;
        desktop_items[num_desktop_items].y = 30 + (num_desktop_items / 10) * 80;
        
        if (file_list[i].attr & FAT_ATTR_DIRECTORY) {
            desktop_items[num_desktop_items].type = ICON_DIR;
        } else {
            desktop_items[num_desktop_items].type = ICON_FILE;
        }
        num_desktop_items++;
    }
}

    void add_window(Window* win) {
        if (num_windows < 16) {
            if (focused_idx != -1 && focused_idx < num_windows) windows[focused_idx]->has_focus = false;
            windows[num_windows] = win;
            focused_idx = num_windows;
            windows[num_windows]->has_focus = true;
            num_windows++;
        }
    }

    void set_focus(int idx) {
        if (idx < 0 || idx >= num_windows || idx == focused_idx) return;
        if (focused_idx != -1 && focused_idx < num_windows) windows[focused_idx]->has_focus = false;
        Window* focused = windows[idx];
        for (int i = idx; i < num_windows - 1; i++) windows[i] = windows[i+1];
        windows[num_windows - 1] = focused;
        focused_idx = num_windows - 1;
        windows[num_windows - 1]->has_focus = true;
    }

    int get_num_windows() const { return num_windows; }
    int get_focused_idx() const { return focused_idx; }
    Window* get_window(int idx) { 
        if (idx >= 0 && idx < num_windows) return windows[idx];
        return nullptr;
    }

    // Which ELF slot (if any) does the currently FOCUSED window own?
    // -1 if no window is focused, or the focused window isn't a
    // terminal capturing an ELF process. Used to route keystrokes to
    // whichever process the user actually clicked into, instead of
    // just the first process that happens to be waiting for input.
    int get_focused_elf_slot() const {
        if (focused_idx < 0 || focused_idx >= num_windows) return -1;
        return windows[focused_idx]->get_elf_slot();
    }

    void cleanup_closed_windows() {
        if (num_windows == 0) return;
        int current_idx = 0;
        while (current_idx < num_windows) {
            if (windows[current_idx]->is_closed) {
                delete windows[current_idx];
                for (int j = current_idx; j < num_windows - 1; j++) {
                    windows[j] = windows[j + 1];
                }
                num_windows--;
            } else {
                current_idx++;
            }
        }
        
        if (num_windows > 0) {
            focused_idx = num_windows - 1;
            for(int i = 0; i < num_windows; i++) windows[i]->has_focus = false;
            windows[focused_idx]->has_focus = true;
        } else {
            focused_idx = -1;
        }
    }

    // Helper: draw a single 3D-style taskbar button
    void draw_taskbar_button(int bx, int by, int bw, int bh,
                             const char* label, bool active) {
        using namespace ColorPalette;
        uint32_t face  = active ? 0x6080C0u : BUTTON_FACE;
        uint32_t tcolor = active ? TEXT_WHITE : TEXT_BLACK;
        draw_rect_filled(bx, by, bw, 1,  BUTTON_HIGHLIGHT);
        draw_rect_filled(bx, by, 1,  bh, BUTTON_HIGHLIGHT);
        draw_rect_filled(bx + 1, by + bh - 1, bw - 1, 1, BUTTON_SHADOW);
        draw_rect_filled(bx + bw - 1, by + 1,  1, bh - 1, BUTTON_SHADOW);
        draw_rect_filled(bx + 1, by + 1, bw - 2, bh - 2, face);
        // Truncate label to fit
        char tmp[16];
        int maxc = (bw - 8) / 8;
        if (maxc < 1) maxc = 1;
        if (maxc > 15) maxc = 15;
        int li = 0;
        for (; li < maxc && label[li]; li++) tmp[li] = label[li];
        tmp[li] = 0;
        draw_string(tmp, bx + 4, by + bh/2 - 4, tcolor);
    }

    void draw_desktop() {
        using namespace ColorPalette;
        
        // Taskbar base
        draw_rect_filled(0, fb_info.height - 40, fb_info.width, 40, TASKBAR_GRAY);
        draw_rect_filled(0, fb_info.height - 40, fb_info.width, 1, BUTTON_HIGHLIGHT);
        
        // ── "Terminal" launcher button (always present, left-most) ──────────
        int btn_y  = fb_info.height - 36;
        int btn_h  = 32;
        int btn_x  = 4;
        int btn_w  = 80;
        draw_taskbar_button(btn_x, btn_y, btn_w, btn_h, "Terminal", false);
        btn_x += btn_w + 4;

        // ── Per-slot ELF process buttons ─────────────────────────────────────
        // Each active ELF slot gets its own button. Clicking it raises and
        // focuses the terminal window that owns the slot. The active (focused)
        // slot button is drawn highlighted in blue.
        for (int s = 0; s < MAX_ELF_PROCESSES; ++s) {
            if (!elf_processes[s].active) continue;  // skip idle and completed slots
            // Build a label: "Slot N: <first few chars of cmdline>"
            char label[20];
            label[0] = 'S'; label[1] = '0' + (char)s; label[2] = ':'; label[3] = ' ';
            // Append the first few chars of cmdline (or "elf" if empty)
            const char* cmd = elf_processes[s].cmdline;
            int ci = 4;
            if (cmd[0]) {
                for (int k = 0; cmd[k] && ci < 15; k++, ci++) label[ci] = cmd[k];
            } else {
                label[ci++] = 'e'; label[ci++] = 'l'; label[ci++] = 'f';
            }
            label[ci] = 0;

            // Is this slot's terminal the focused window?
            bool is_focused = false;
            if (focused_idx >= 0 && focused_idx < num_windows) {
                if (windows[focused_idx]->get_elf_slot() == s) is_focused = true;
            }

            draw_taskbar_button(btn_x, btn_y, btn_w, btn_h, label, is_focused);
            btn_x += btn_w + 4;
            if (btn_x + btn_w >= (int)fb_info.width - 100) break; // guard overflow
        }

        // Draw desktop icons
        for (int i = 0; i < num_desktop_items; ++i) {
            bool is_shortcut = strstr(desktop_items[i].name, ".lnk") != nullptr;
            if (desktop_items[i].type == ICON_APP) {
                draw_icon_folder(desktop_items[i].x, desktop_items[i].y);
            } else {
                draw_icon_file(desktop_items[i].x, desktop_items[i].y, is_shortcut);
            }
            draw_string(desktop_items[i].name, desktop_items[i].x, desktop_items[i].y + 35, TEXT_WHITE);
        }
    }

    void execute_context_menu_action(int item_index); // New

    // =============================================================================
    // STATE-BASED WINDOW MANAGER UPDATE - ATOMIC FRAME RENDERING
    // =============================================================================
    void update_all() {
        // Phase 0: Begin new frame
        if (g_render_state.renderPhase == 0) {
            g_render_state.frameComplete = false;
            g_render_state.backgroundCleared = false;
            g_render_state.currentWindow = 0;
            g_render_state.renderPhase = 1;
        }
        
        // Phase 1: Clear background (done once per frame in main loop)
        if (g_render_state.renderPhase == 1) {
            g_render_state.backgroundCleared = true;
            g_render_state.renderPhase = 2;
        }
        
        // Phase 2: Draw desktop and icons
        if (g_render_state.renderPhase == 2) {
            draw_desktop();
            g_render_state.renderPhase = 3;
        }
        
        // Phase 3: Draw windows (all at once to prevent tearing)
        if (g_render_state.renderPhase == 3) {
            for (int i = 0; i < num_windows; i++) {
                if (windows[i] && !windows[i]->is_closed) {
                    windows[i]->draw();
                }
            }
            g_render_state.renderPhase = 4;
        }

        // New Phase 3.5: Draw context menu on top of everything
        if (context_menu_active) {
            int menu_width = 150;
            int item_height = 20;
            int menu_height = num_context_menu_items * item_height;
            draw_rect_filled(context_menu_x, context_menu_y, menu_width, menu_height, ColorPalette::BUTTON_FACE);
            draw_rect_filled(context_menu_x, context_menu_y, menu_width, 1, ColorPalette::BUTTON_HIGHLIGHT);
            draw_rect_filled(context_menu_x, context_menu_y, 1, menu_height, ColorPalette::BUTTON_HIGHLIGHT);
            draw_rect_filled(context_menu_x+menu_width-1, context_menu_y, 1, menu_height, ColorPalette::BUTTON_SHADOW);
            draw_rect_filled(context_menu_x, context_menu_y+menu_height-1, menu_width, 1, ColorPalette::BUTTON_SHADOW);

            for (int i = 0; i < num_context_menu_items; ++i) {
                draw_string(context_menu_items[i], context_menu_x + 5, context_menu_y + 5 + i * item_height, ColorPalette::TEXT_BLACK);
            }
        }
        
        // Phase 4: Update logic
        if (g_render_state.renderPhase == 4) {
            for (int i = 0; i < num_windows; i++) {
                if (windows[i] && !windows[i]->is_closed) {
                    windows[i]->update();
                }
            }
            g_render_state.renderPhase = 5;
        }
        
        // Phase 5: Frame complete
        if (g_render_state.renderPhase == 5) {
            g_render_state.frameComplete = true;
            g_render_state.renderPhase = 0;
            g_render_state.frameNumber++;
        }
    }

    void handle_input(char key, int mx, int my, bool left_down, bool left_clicked, bool right_clicked); // Modified
    void print_to_focused(const char* s);
};

WindowManager wm;


// =============================================================================
// I/O WAIT AND PS/2 FUNCTIONS
// =============================================================================

static inline void io_wait_short() {
    asm volatile("outb %%al, $0x80" : : "a"(0));
}

static inline void io_delay_short() {
    for (volatile int i = 0; i < 1; i++) {
        io_wait_short();
    }
}

static inline void io_delay_medium() {
    for (volatile int i = 0; i < 2; i++) {
        io_wait_short();
    }
}

static inline void io_delay_long() {
    for (volatile int i = 0; i < 10; i++) {
        io_wait_short();
    }
}

static bool ps2_wait_input_ready(uint32_t timeout = 1000) {
    while (timeout--) {
        if (!(inb(PS2_STATUS_PORT) & PS2_STATUS_INPUT_FULL)) {
            return true;
        }
        if (timeout % 1000 == 0) io_delay_medium();
    }
    return false;
}

static bool ps2_wait_output_ready(uint32_t timeout = 1000) {
    while (timeout--) {
        if (inb(PS2_STATUS_PORT) & PS2_STATUS_OUTPUT_FULL) {
            return true;
        }
        if (timeout % 1000 == 0) io_delay_medium();
    }
    return false;
}

static void ps2_flush_output_buffer() {
    int timeout = 10;
    while ((inb(PS2_STATUS_PORT) & PS2_STATUS_OUTPUT_FULL) && timeout--) {
        inb(PS2_DATA_PORT);
        io_delay_medium();
    }
}

static bool ps2_write_command(uint8_t cmd) {
    if (!ps2_wait_input_ready()) return false;
    outb(PS2_COMMAND_PORT, cmd);
    io_delay_medium();
    return true;
}

static bool ps2_write_data(uint8_t data) {
    if (!ps2_wait_input_ready()) return false;
    outb(PS2_DATA_PORT, data);
    io_delay_medium();
    return true;
}

static bool ps2_read_data(uint8_t* data) {
    if (!ps2_wait_output_ready()) return false;
    *data = inb(PS2_DATA_PORT);
    return true;
}

static bool ps2_mouse_write_command(uint8_t cmd, int max_retries = 3) {
    for (int retry = 0; retry < max_retries; retry++) {
        if (!ps2_write_command(PS2_CMD_WRITE_PORT2)) continue;
        if (!ps2_write_data(cmd)) continue;
        
        uint8_t response;
        if (ps2_read_data(&response)) {
            if (response == PS2_ACK) {
                return true;
            } else if (response == PS2_RESEND) {
                io_delay_long();
                continue;
            }
        }
        io_delay_long();
    }
    return false;
}

static bool ps2_mouse_write_with_arg(uint8_t cmd, uint8_t arg) {
    if (!ps2_mouse_write_command(cmd)) return false;
    io_delay_medium();
    return ps2_mouse_write_command(arg);
}

static bool init_ps2_mouse_legacy() {
    outb(0x64, 0xA8);
    io_delay_long();
    
    outb(0x64, 0x20);
    uint8_t status = inb(0x60) | 2;
    status &= ~0x20;
    
    outb(0x64, 0x60);
    outb(0x60, status);
    io_delay_long();
    
    outb(0x64, 0xD4);
    outb(0x60, 0xF6);
    inb(0x60);
    io_delay_long();
    
    outb(0x64, 0xD4);
    outb(0x60, 0xF4);
    inb(0x60);
    io_delay_long();
    
    ps2_flush_output_buffer();
    return true;
}

static inline void pci_write_config_dword(uint16_t bus, uint8_t device, uint8_t function, uint8_t offset, uint32_t value) {
    uint32_t address = 0x80000000 | ((uint32_t)bus << 16) | ((uint32_t)device << 11) | ((uint32_t)function << 8) | (offset & 0xFC);
    outl(0xCF8, address);
    outl(0xCFC, value);
}

struct USBLegacyInfo {
    bool has_uhci;
    bool has_ehci;
    bool has_xhci;
    uint64_t legacy_base;
    bool ps2_emulation_active;
    uint16_t pci_bus;
    uint8_t pci_device;
    uint8_t pci_function;
};

static USBLegacyInfo usb_info = {false, false, false, 0, false, 0, 0, 0};

static bool detect_usb_controllers() {
    for (uint16_t bus = 0; bus < 8; bus++) {  /* scan 8 buses: covers real HW and QEMU */
        for (uint8_t device = 0; device < 32; device++) {
            uint32_t vid = pci_read_config_dword(bus, device, 0, 0) & 0xFFFF;
            if (vid == 0xFFFF) continue;  /* slot empty */
            uint32_t class_code = pci_read_config_dword(bus, device, 0, 0x08);
            uint8_t base_class = (class_code >> 24) & 0xFF;
            uint8_t sub_class = (class_code >> 16) & 0xFF;
            uint8_t prog_if = (class_code >> 8) & 0xFF;
            
            if (base_class == 0x0C && sub_class == 0x03) {
                if (prog_if == 0x20) usb_info.has_ehci = true;
                else if (prog_if == 0x30) usb_info.has_xhci = true;
                
                usb_info.pci_bus = bus;
                usb_info.pci_device = device;
                usb_info.pci_function = 0;
                
                uint32_t bar0 = pci_read_config_dword(bus, device, 0, 0x10);
                usb_info.legacy_base = bar0 & 0xFFFFFFF0;
                return true;
            }
        }
    }
    return false;
}

static bool enable_usb_legacy_support() {
    if (usb_info.has_ehci) {
        uint32_t hccparams = pci_read_config_dword(
            usb_info.pci_bus, 
            usb_info.pci_device, 
            usb_info.pci_function, 
            0x08
        );
        
        uint8_t eecp = (hccparams >> 8) & 0xFF;
        
        if (eecp >= 0x40) {
            uint32_t legsup = pci_read_config_dword(
                usb_info.pci_bus, 
                usb_info.pci_device, 
                usb_info.pci_function, 
                eecp
            );
            
            legsup |= (1 << 24);
            pci_write_config_dword(
                usb_info.pci_bus, 
                usb_info.pci_device, 
                usb_info.pci_function, 
                eecp, 
                legsup
            );
            
            for (int i = 0; i < 100; i++) {
                io_delay_long();
                legsup = pci_read_config_dword(
                    usb_info.pci_bus, 
                    usb_info.pci_device, 
                    usb_info.pci_function, 
                    eecp
                );
                if (!(legsup & (1 << 16))) break;
            }
            
            uint32_t usblegctlsts = pci_read_config_dword(
                usb_info.pci_bus, 
                usb_info.pci_device, 
                usb_info.pci_function, 
                eecp + 4
            );
            usblegctlsts &= 0xFFFF0000;
            pci_write_config_dword(
                usb_info.pci_bus, 
                usb_info.pci_device, 
                usb_info.pci_function, 
                eecp + 4, 
                usblegctlsts
            );
            
            return true;
        }
    }
    return false;
}

static bool init_ps2_mouse_hardware() {
    uint8_t data;
    
    if (usb_info.ps2_emulation_active) {
        io_delay_long();
    }
    
    ps2_write_command(PS2_CMD_DISABLE_PORT1);
    io_delay_long();
    ps2_write_command(PS2_CMD_DISABLE_PORT2);
    io_delay_long();
    
    for (int i = 0; i < 16; i++) {
        if (inb(PS2_STATUS_PORT) & PS2_STATUS_OUTPUT_FULL) {
            inb(PS2_DATA_PORT);
        }
        io_delay_medium();
    }
    
    if (!ps2_write_command(PS2_CMD_TEST_CTRL)) return false;
    io_delay_long();
    
    bool self_test_passed = false;
    for (int retry = 0; retry < 5; retry++) {
        if (ps2_read_data(&data)) {
            if (data == 0x55) {
                self_test_passed = true;
                break;
            }
        }
        io_delay_long();
    }
    
    if (!self_test_passed) {
        return false;
    }
    
    if (!ps2_write_command(PS2_CMD_READ_CONFIG)) return false;
    if (!ps2_read_data(&data)) return false;
    
    uint8_t config = data;
    config |= 0x03;
    config &= ~0x30;
    
    if (!ps2_write_command(PS2_CMD_WRITE_CONFIG)) return false;
    if (!ps2_write_data(config)) return false;
    io_delay_long();
    
    if (!ps2_write_command(PS2_CMD_TEST_PORT2)) return false;
    io_delay_long();
    
    bool port_test_passed = false;
    if (ps2_read_data(&data)) {
        if (data == 0x00) {
            port_test_passed = true;
        }
    }
    
    if (!port_test_passed) {
        return false;
    }
    
    if (!ps2_write_command(PS2_CMD_ENABLE_PORT2)) return false;
    io_delay_long();
    
    if (!ps2_mouse_write_command(MOUSE_CMD_RESET)) return false;
    
    uint32_t bat_timeout = 500;
    bool bat_complete = false;
    
    while (bat_timeout-- > 0) {
        if (ps2_read_data(&data)) {
            if (data == 0xAA) {
                bat_complete = true;
                io_delay_medium();
                ps2_read_data(&data);
                break;
            } else if (data == 0xFC) {
                io_delay_long();
                ps2_mouse_write_command(MOUSE_CMD_RESET);
                bat_timeout = 250;
            }
        }
        if (bat_timeout % 100 == 0) {
            io_delay_medium();
        }
    }
    
    if (!bat_complete) {
        return false;
    }
    
    io_delay_long();
    
    if (!ps2_mouse_write_command(MOUSE_CMD_SET_DEFAULTS)) return false;
    io_delay_long();
    
    if (!ps2_mouse_write_with_arg(MOUSE_CMD_SET_SAMPLE, 100)) {
    }
    io_delay_long();
    
    if (!ps2_mouse_write_with_arg(MOUSE_CMD_SET_RESOLUTION, 3)) {
    }
    io_delay_long();
    
    outb(0x64, 0xD4);
    io_delay_medium();
    outb(0x60, 0xE6);
    io_delay_medium();
    inb(0x60);
    io_delay_medium();
    
    if (!ps2_mouse_write_command(MOUSE_CMD_ENABLE_DATA)) return false;
    io_delay_long();
    
    ps2_write_command(PS2_CMD_ENABLE_PORT1);
    io_delay_long();
    
    for (int i = 0; i < 16; i++) {
        if (inb(PS2_STATUS_PORT) & PS2_STATUS_OUTPUT_FULL) {
            inb(PS2_DATA_PORT);
        }
        io_delay_short();
    }
    
    return true;
}

bool initialize_universal_mouse() {
    universal_mouse_state.initialized = false;
    universal_mouse_state.synchronized = false;
    universal_mouse_state.packet_cycle = 0;
    universal_mouse_state.x = fb_info.width / 2;
    universal_mouse_state.y = fb_info.height / 2;
    
    bool has_usb = detect_usb_controllers();
    if (has_usb) {
        wm.print_to_focused("USB controllers detected...\n");
        if (enable_usb_legacy_support()) {
            wm.print_to_focused("USB Legacy PS/2 emulation enabled.\n");
        }
    }
    
    wm.print_to_focused("Initializing PS/2 mouse interface...\n");
    
    if (init_ps2_mouse_hardware()) {
        universal_mouse_state.initialized = true;
        wm.print_to_focused("PS/2 mouse initialized (hardware method).\n");
        return true;
    }
    
    wm.print_to_focused("Trying legacy PS/2 initialization...\n");
    if (init_ps2_mouse_legacy()) {
        universal_mouse_state.initialized = true;
        wm.print_to_focused("PS/2 mouse initialized (legacy method).\n");
        return true;
    }
    
    wm.print_to_focused("ERROR: Mouse initialization failed.\n");
    return false;
}
void poll_input_universal() {
    last_key_press = 0;
    // Non-blocking: only read if data is immediately available

    for (int iterations = 0; iterations < 16; iterations++) {
        uint8_t status = inb(PS2_STATUS_PORT);
        if (!(status & PS2_STATUS_OUTPUT_FULL)) break;

        uint8_t data = inb(PS2_DATA_PORT);

        if (status & PS2_STATUS_AUX_DATA) {
            process_universal_mouse_packet(data);
        } else {
            bool is_press = !(data & 0x80);
            uint8_t scancode = data & 0x7F;

            if (scancode == 0 || scancode > 0x58) continue;

            if (scancode == 0x2A || scancode == 0x36) {
                is_shift_pressed = is_press;
            } else if (scancode == 0x1D) {
                is_ctrl_pressed = is_press;
            } else if (is_press) {
                switch(scancode) {
                    case 0x48: last_key_press = KEY_UP; break;
                    case 0x50: last_key_press = KEY_DOWN; break;
                    case 0x4B: last_key_press = KEY_LEFT; break;
                    case 0x4D: last_key_press = KEY_RIGHT; break;
                    case 0x53: last_key_press = KEY_DELETE; break;
                    case 0x47: last_key_press = KEY_HOME; break;
                    case 0x4F: last_key_press = KEY_END; break;
                    default: {
                        const char* map = is_ctrl_pressed ? sc_ascii_ctrl_map :
                                          (is_shift_pressed ? sc_ascii_shift_map : sc_ascii_nomod_map);
                        if (scancode < 128 && map[scancode] != 0) {
                            last_key_press = map[scancode];
                        }
                    }
                }
            }
        }
    }

    mouse_x = universal_mouse_state.x;
    mouse_y = universal_mouse_state.y;
    mouse_left_down = universal_mouse_state.left_button;
    mouse_right_down = universal_mouse_state.right_button; // New
}
void draw_cursor(int x, int y, uint32_t color) { 
    for(int i=0;i<12;i++) put_pixel_back(x,y+i,color); 
    for(int i=0;i<8;i++) put_pixel_back(x+i,y+i,color); 
    for(int i=0;i<4;i++) put_pixel_back(x+i,y+(11-i),color); 
}




// =============================================================================
// SECTION 5: DISK DRIVER & FAT32 FILESYSTEM
// =============================================================================
#define SATA_SIG_ATA 0x00000101
#define PORT_CMD_ST 0x00000001
#define PORT_CMD_FRE 0x00000010
#define ATA_CMD_READ_DMA_EXT 0x25
#define ATA_CMD_WRITE_DMA_EXT 0x35
#define HBA_PORT_CMD_CR 0x00008000
#define TFD_STS_BSY 0x80
#define TFD_STS_DRQ 0x08
#define FIS_TYPE_REG_H2D 0x27
#define DELETED_ENTRY 0xE5
#define ATTR_LONG_NAME 0x0F
#define ATTR_VOLUME_ID 0x08
#define ATTR_ARCHIVE 0x20
#define FAT_FREE_CLUSTER 0x00000000
#define FAT_END_OF_CHAIN 0x0FFFFFFF

// =============================================================================
// FILE EXPLORER WINDOW IMPLEMENTATION (New)
// =============================================================================



// Add these definitions near the other AHCI/FAT32 structs
typedef volatile struct {
    uint32_t clb;         // 0x00, command list base address, 1K-byte aligned
    uint32_t clbu;        // 0x04, command list base address upper 32 bits
    uint32_t fb;          // 0x08, FIS base address, 256-byte aligned
    uint32_t fbu;         // 0x0C, FIS base address upper 32 bits
    uint32_t is;          // 0x10, interrupt status
    uint32_t ie;          // 0x14, interrupt enable
    uint32_t cmd;         // 0x18, command and status
    uint32_t rsv0;        // 0x1C, Reserved
    uint32_t tfd;         // 0x20, task file data
    uint32_t sig;         // 0x24, signature
    uint32_t ssts;        // 0x28, SATA status (SCR0:SStatus)
    uint32_t sctl;        // 0x2C, SATA control (SCR2:SControl)
    uint32_t serr;        // 0x30, SATA error (SCR1:SError)
    uint32_t sact;        // 0x34, SATA active (SCR3:SActive)
    uint32_t ci;          // 0x38, command issue
    uint32_t sntf;        // 0x3C, SATA notification (SCR4:SNotification)
    uint32_t fbs;         // 0x40, FIS-based switching control
    uint32_t rsv1[11];    // 0x44 ~ 0x6F, Reserved
    uint32_t vendor[4];   // 0x70 ~ 0x7F, vendor specific
} HBA_PORT;

typedef volatile struct {
    uint32_t cap;         // 0x00, Host capability
    uint32_t ghc;         // 0x04, Global host control
    uint32_t is;          // 0x08, Interrupt status
    uint32_t pi;          // 0x0C, Port implemented
    uint32_t vs;          // 0x10, Version
    uint32_t ccc_ctl;     // 0x14, Command completion coalescing control
    uint32_t ccc_pts;     // 0x18, Command completion coalescing ports
    uint32_t em_loc;      // 0x1C, Enclosure management location
    uint32_t em_ctl;      // 0x20, Enclosure management control
    uint32_t cap2;        // 0x24, Host capabilities extended
    uint32_t bohc;        // 0x28, BIOS/OS handoff control and status
    uint8_t  rsv[0x60-0x2C];
    uint8_t  vendor[0x90-0x60]; // Vendor specific registers
    HBA_PORT ports[1];    // 0x90 ~ HBA memory mapped space, 1 ~ 32 ports
} HBA_MEM;


typedef struct { 
    uint8_t order; 
    uint16_t name1[5]; 
    uint8_t attr; 
    uint8_t type; 
    uint8_t checksum; 
    uint16_t name2[6]; 
    uint16_t fst_clus_lo; 
    uint16_t name3[2]; 
} __attribute__((packed)) fat_lfn_entry_t;

uint8_t lfn_checksum(const unsigned char *p_fname) {
    uint8_t sum = 0;
    for (int i = 11; i; i--) {
        sum = ((sum & 1) ? 0x80 : 0) + (sum >> 1) + *p_fname++;
    }
    return sum;
}

static int g_ahci_port = -1; // Will store the first active port number
static int g_selected_port = -1; // User-selected disk port (-1 = use g_ahci_port)
static bool g_disk_unlocked = false;
static char g_disk_password_file[] = ".diskpass";
typedef struct { uint8_t cfl:5, a:1, w:1, p:1, r:1, b:1, c:1, res0:1; uint16_t prdtl; volatile uint32_t prdbc; uint64_t ctba; uint32_t res1[4]; } __attribute__((packed)) HBA_CMD_HEADER;
typedef struct { uint64_t dba; uint32_t res0; uint32_t dbc:22, res1:9, i:1; } __attribute__((packed)) HBA_PRDT_ENTRY;
typedef struct { uint8_t fis_type, pmport:4, res0:3, c:1, command, featurel; uint8_t lba0, lba1, lba2, device; uint8_t lba3, lba4, lba5, featureh; uint8_t countl, counth, icc, control; uint8_t res1[4]; } __attribute__((packed)) FIS_REG_H2D;
typedef struct { uint8_t jmp[3]; char oem[8]; uint16_t bytes_per_sec; uint8_t sec_per_clus; uint16_t rsvd_sec_cnt; uint8_t num_fats; uint16_t root_ent_cnt; uint16_t tot_sec16; uint8_t media; uint16_t fat_sz16; uint16_t sec_per_trk; uint16_t num_heads; uint32_t hidd_sec; uint32_t tot_sec32; uint32_t fat_sz32; uint16_t ext_flags; uint16_t fs_ver; uint32_t root_clus; uint16_t fs_info; uint16_t bk_boot_sec; uint8_t res[12]; uint8_t drv_num; uint8_t res1; uint8_t boot_sig; uint32_t vol_id; char vol_lab[11]; char fil_sys_type[8]; } __attribute__((packed)) fat32_bpb_t;


static uint64_t ahci_base = 0;
static HBA_CMD_HEADER* cmd_list;
static char* cmd_table_buffer;
// FIS receive buffer. Promoted from a disk_init() local to a global so
// ahci_port_setup() can program PxFB/PxFBU for any port, not just the
// one disk_init() auto-selected.
static char* g_ahci_fis_buffer = nullptr;
static fat32_bpb_t bpb;
static uint32_t fat_start_sector, data_start_sector;
static uint32_t current_directory_cluster = 0;
// Absolute disk LBA of the FAT32 partition's first sector (0 if the disk
// is a raw FAT32 image with no partition table — e.g. mkfat32.py output).
// Set by fat32_init when an MBR/GPT FAT32 partition is located; used by
// fat32_format so a format command on a partitioned bare-metal disk
// rewrites the partition's BPB instead of clobbering the MBR.
static uint64_t g_partition_lba = 0;

// --- Aligned Memory Allocator ---
void* alloc_aligned(size_t size, size_t alignment) {
    size_t offset = alignment - 1 + sizeof(void*);
    void* p1 = operator new(size + offset);
    if (p1 == nullptr) return nullptr;
    void** p2 = (void**)(((uintptr_t)p1 + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void free_aligned(void* ptr) {
    if (ptr == nullptr) return;
    operator delete(((void**)ptr)[-1]);
}
// Forward declarations: stop_cmd / start_cmd are defined further down
// (after read_write_sectors) but ahci_port_setup below needs them.
void stop_cmd(HBA_PORT* port);
void start_cmd(HBA_PORT* port);

// Program one AHCI port's command-list / FIS base registers and start
// its command engine. disk_init() did this inline for the single port
// it auto-selected; select_disk could then switch g_ahci_port to a
// DIFFERENT implemented port whose clb/fb were never programmed, so
// every subsequent command against it stalled (port->ci never cleared)
// and all I/O failed. Both callers now go through this so any port the
// kernel talks to is always fully initialised first.
//
// Returns true if the port has a device present and was set up.
static bool ahci_port_setup(int port_index) {
    if (!ahci_base || port_index < 0 || port_index >= 32) return false;
    if (!cmd_list || !cmd_table_buffer) return false;

    HBA_PORT* port = (HBA_PORT*)(ahci_base + 0x100 + (port_index * 0x80));

    uint8_t det = port->ssts & 0x0F;
    uint8_t ipm = (port->ssts >> 8) & 0x0F;
    if (det != 3 || ipm != 1) return false;     // no active device

    stop_cmd(port);

    port->clb  = (uint32_t)(uintptr_t)cmd_list;
    port->clbu = (uint32_t)(((uint64_t)(uintptr_t)cmd_list) >> 32);
    port->fb   = (uint32_t)(uintptr_t)g_ahci_fis_buffer;
    port->fbu  = (uint32_t)(((uint64_t)(uintptr_t)g_ahci_fis_buffer) >> 32);

    port->serr = 0xFFFFFFFF;
    port->is   = 0xFFFFFFFF;                    // clear stale interrupts

    start_cmd(port);
    return true;
}

void cmd_list_and_select_disk(const char* arg) {
    // List all detected AHCI ports
    if (!ahci_base) {
        wm.print_to_focused("No AHCI controller found.\n");
        return;
    }

    uint32_t ports_implemented = *(volatile uint32_t*)(ahci_base + 0x0C);
    char msg[128];

    if (!arg || arg[0] == '\0') {
        // No argument: list available disks
        wm.print_to_focused("Available disks:\n");
        bool found = false;
        for (int i = 0; i < 32; i++) {
            if (!(ports_implemented & (1 << i))) continue;
            HBA_PORT* port = (HBA_PORT*)(ahci_base + 0x100 + (i * 0x80));
            uint8_t det = port->ssts & 0x0F;
            uint8_t ipm = (port->ssts >> 8) & 0x0F;
            if (det != 3 || ipm != 1) continue;

            int active = (i == g_ahci_port) ? 1 : 0;
            int selected = (i == g_selected_port || (g_selected_port == -1 && i == g_ahci_port)) ? 1 : 0;

            snprintf(msg, 128, "  Port %d: %s%s\n",
                     i,
                     active  ? "[AHCI] " : "",
                     selected ? "<-- selected" : "");
            wm.print_to_focused(msg);
            found = true;
        }
        if (!found) wm.print_to_focused("  (no drives detected)\n");
        wm.print_to_focused("Usage: select_disk <port>\n");
        return;
    }

    // Argument given: select that port
    int requested = simple_atoi(arg);
    if (!(ports_implemented & (1 << requested))) {
        snprintf(msg, 128, "Port %d not implemented.\n", requested);
        wm.print_to_focused(msg);
        return;
    }

    HBA_PORT* port = (HBA_PORT*)(ahci_base + 0x100 + (requested * 0x80));
    uint8_t det = port->ssts & 0x0F;
    uint8_t ipm = (port->ssts >> 8) & 0x0F;
    if (det != 3 || ipm != 1) {
        snprintf(msg, 128, "Port %d has no active drive (det=%d ipm=%d).\n", requested, det, ipm);
        wm.print_to_focused(msg);
        return;
    }

    // Switch disk. Program the target port's command-list / FIS base
    // and start its command engine BEFORE issuing any I/O to it —
    // disk_init() only set up the port it auto-selected, so without this
    // a switch to any other port left it uninitialised and every read
    // or write against it stalled.
    if (!ahci_port_setup(requested)) {
        snprintf(msg, 128, "Port %d setup failed.\n", requested);
        wm.print_to_focused(msg);
        return;
    }
    g_selected_port = requested;
    g_ahci_port     = requested;           // redirect all I/O immediately

    // Re-initialise FAT32 on the new disk
    bool ok = fat32_init();
    snprintf(msg, 128, "Switched to disk port %d. FAT32: %s\n",
             requested, ok ? "OK" : "not found / failed");
    wm.print_to_focused(msg);

    if (ok) wm.load_desktop_items();      // refresh desktop icons from new disk
}int read_write_sectors(int port_num, uint64_t lba, uint16_t count,
                       bool write, void* buffer) {
    if (port_num < 0 || port_num >= 32 || !ahci_base) return -1;

    HBA_PORT* port = (HBA_PORT*)(ahci_base + 0x100 + (port_num * 0x80));
    port->is = 0xFFFFFFFF;

    uint32_t slots = (port->sact | port->ci);
    int slot = -1;
    for (int i = 0; i < 32; i++) {
        if ((slots & (1 << i)) == 0) { slot = i; break; }
    }
    if (slot == -1) return -1;

    // --- WRITE PATH: encrypt a copy before sending to disk ---
    uint8_t* enc_buf = nullptr;
    void*    io_buf  = buffer;

    if (write && g_fs_encryption_enabled) {
        enc_buf = new uint8_t[count * SECTOR_SIZE];
        if (!enc_buf) return -1;
        memcpy(enc_buf, buffer, count * SECTOR_SIZE);
        for (int s = 0; s < count; s++) {
            xor_sector(enc_buf + s * SECTOR_SIZE, lba + s);
        }
        io_buf = enc_buf;
    }

    HBA_CMD_HEADER* cmd_header = &cmd_list[slot];
    cmd_header->cfl    = sizeof(FIS_REG_H2D) / sizeof(uint32_t);
    cmd_header->w      = write;
    cmd_header->prdtl  = 1;

    uintptr_t       cmd_table_addr = (uintptr_t)cmd_header->ctba;
    FIS_REG_H2D*    cmd_fis        = (FIS_REG_H2D*)(cmd_table_addr);
    HBA_PRDT_ENTRY* prdt           = (HBA_PRDT_ENTRY*)(cmd_table_addr + 128);

    prdt->dba = (uint64_t)(uintptr_t)io_buf;
    prdt->dbc = (count * SECTOR_SIZE) - 1;
    prdt->i   = 0;

    memset(cmd_fis, 0, sizeof(FIS_REG_H2D));
    cmd_fis->fis_type = FIS_TYPE_REG_H2D;
    cmd_fis->c        = 1;
    cmd_fis->command  = write ? ATA_CMD_WRITE_DMA_EXT : ATA_CMD_READ_DMA_EXT;
    cmd_fis->lba0     = (uint8_t)lba;
    cmd_fis->lba1     = (uint8_t)(lba >> 8);
    cmd_fis->lba2     = (uint8_t)(lba >> 16);
    cmd_fis->device   = 1 << 6;
    cmd_fis->lba3     = (uint8_t)(lba >> 24);
    cmd_fis->lba4     = (uint8_t)(lba >> 32);
    cmd_fis->lba5     = (uint8_t)(lba >> 40);
    cmd_fis->countl   = count & 0xFF;
    cmd_fis->counth   = (count >> 8) & 0xFF;

    while (port->tfd & (TFD_STS_BSY | TFD_STS_DRQ));
    port->ci = (1 << slot);

    // Wait for the command slot to clear. The previous budget (100000
    // tight-loop iterations) was far too small: a real DMA write —
    // especially the first WRITE_DMA_EXT after the command engine has
    // been idle, e.g. the boot-sector write in formatfs — routinely did
    // not finish within it, so the function returned -1 and the format
    // reported "Failed to write new boot sector". Reads happened to fit
    // the old budget often enough to look reliable. Use a much larger
    // budget and ALSO bail out early on a real task-file error rather
    // than relying on the spin count alone.
    const long IO_TIMEOUT = 200000000L;   // generous; covers slow writes
    long spin = 0;
    bool timed_out = false;
    while (true) {
        if ((port->ci & (1 << slot)) == 0) break;   // command finished
        if (port->is & (1 << 30)) break;            // TFES: task-file error
        if (++spin >= IO_TIMEOUT) { timed_out = true; break; }
    }

    if (enc_buf) { delete[] enc_buf; enc_buf = nullptr; }

    if (timed_out) return -1;
    // Real error if the TFES interrupt fired OR the task-file register
    // reports ERR (bit 0). PxTFD.STS bit0 = ERR; checking it catches a
    // device-rejected command even when PxIS bit 30 was already cleared.
    if (port->is & (1 << 30)) return -1;
    if (port->tfd & 0x01)     return -1;

    // --- READ PATH: decrypt in place after receiving from disk ---
    if (!write && g_fs_encryption_enabled) {
        for (int s = 0; s < count; s++) {
            xor_sector((uint8_t*)buffer + s * SECTOR_SIZE, lba + s);
        }
    }

    return 0;
}
void stop_cmd(HBA_PORT *port) {
    port->cmd &= ~0x0001; // Clear ST (Start)
    port->cmd &= ~0x0010; // Clear FRE (FIS Receive Enable)

    // Wait until Command List Running (CR) and FIS Receive Running (FR) are cleared
    while(port->cmd & 0x8000 || port->cmd & 0x4000);
}

// Helper to start a port's command engine
void start_cmd(HBA_PORT *port) {
    // Wait until Command List Running (CR) is cleared
    while(port->cmd & 0x8000);

    port->cmd |= 0x0010; // Set FRE (FIS Receive Enable)
    port->cmd |= 0x0001; // Set ST (Start)
}
void disk_init() {
    // ─────────────────────────────────────────────────────────────────────
    // Find the AHCI controller on PCI.
    //
    // Why the previous one-liner missed real hardware:
    //   1. It only checked function 0. On every Intel chipset the SATA
    //      controller lives at 00:1F.2 — bus 0, dev 0x1F, function 2.
    //      QEMU and VMware happened to put theirs on function 0, which
    //      is why those worked while bare metal never did.
    //   2. It only matched class/subclass 0x0106 (SATA/AHCI). Many
    //      consumer machines (Dell, HP, Lenovo) ship with BIOS default
    //      "RAID On", in which case the same hardware reports 0x0104
    //      (RAID) while still being AHCI-compatible underneath. Some
    //      Marvell/ASMedia add-in cards report 0x0180 ("Other").
    //   3. It scanned only 8 buses. Cheap to widen.
    // ─────────────────────────────────────────────────────────────────────
    ahci_base = 0;
    uint16_t found_bus = 0; uint8_t found_dev = 0; uint8_t found_fn = 0;
    bool found = false;

    for (uint16_t bus = 0; bus < 256 && !found; bus++) {
        for (uint8_t dev = 0; dev < 32 && !found; dev++) {
            for (uint8_t fn = 0; fn < 8 && !found; fn++) {
                uint32_t vd = pci_read_config_dword(bus, dev, fn, 0x00);
                if ((vd & 0xFFFFu) == 0xFFFFu) continue;   // empty slot

                uint32_t cc = pci_read_config_dword(bus, dev, fn, 0x08);
                uint8_t base_class = (cc >> 24) & 0xFFu;
                uint8_t subclass   = (cc >> 16) & 0xFFu;

                if (base_class != 0x01) continue;          // not mass storage
                // Accept SATA(6), RAID(4), and Other(0x80).
                if (subclass != 0x06 && subclass != 0x04 && subclass != 0x80)
                    continue;

                // BAR5 = ABAR (AHCI Base Memory Register).
                uint32_t bar5 = pci_read_config_dword(bus, dev, fn, 0x24);
                if (bar5 & 1u) continue;                    // I/O BAR, not MMIO
                uint32_t abar = bar5 & 0xFFFFFFF0u;
                if (abar < 0x1000u) continue;               // empty / unmapped

                ahci_base = abar;
                found_bus = bus; found_dev = dev; found_fn = fn;
                found = true;
            }
        }
    }

    if (!ahci_base) {
        wm.print_to_focused("AHCI: no controller found on any PCI bus.\n");
        wm.print_to_focused("  On bare metal: set SATA mode to AHCI in BIOS\n");
        wm.print_to_focused("  (look for 'SATA Operation' / 'SATA Mode Selection').\n");
        return;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Enable bus-master + memory-space decode in the PCI command register.
    // Firmware *usually* leaves these on for the boot device, but UEFI
    // platforms that booted via NVMe/USB sometimes leave the SATA
    // controller un-enabled.
    // ─────────────────────────────────────────────────────────────────────
    {
        uint32_t cmd = pci_read_config_dword(found_bus, found_dev, found_fn, 0x04);
        uint32_t addr_reg = 0x80000000u
                          | ((uint32_t)found_bus << 16)
                          | ((uint32_t)found_dev << 11)
                          | ((uint32_t)found_fn  <<  8)
                          | 0x04u;
        outl(0xCF8, addr_reg);
        outl(0xCFC, cmd | 0x06u);    // bit 1 = memory, bit 2 = bus master
    }

    // ─────────────────────────────────────────────────────────────────────
    // Engage AHCI mode (GHC.AE = bit 31 of Global Host Control at MMIO
    // offset 0x04). Needed when the HBA came up in legacy / IDE-compat
    // mode, which is common on Intel chipsets where the firmware didn't
    // explicitly flip the mode-select bit during POST.
    // ─────────────────────────────────────────────────────────────────────
    {
        volatile uint32_t* ghc = (volatile uint32_t*)(uintptr_t)(ahci_base + 0x04);
        *ghc |= (1u << 31);                                  // AE
        // Brief spin so the HBA acknowledges before we read PxSSTS / PI.
        for (volatile uint32_t i = 0; i < 100000u; i++);
    }

    {
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "AHCI: found at %02x:%02x.%x  ABAR=0x%08x\n",
                 (unsigned)found_bus, (unsigned)found_dev, (unsigned)found_fn,
                 (unsigned)ahci_base);
        wm.print_to_focused(msg);
    }

    // ── Allocate command list / FIS / cmd-table buffers ─────────────────
    cmd_list = (HBA_CMD_HEADER*)alloc_aligned(32 * sizeof(HBA_CMD_HEADER), 1024);
    cmd_table_buffer = (char*)alloc_aligned(32 * 256, 128);
    g_ahci_fis_buffer = (char*)alloc_aligned(256, 256);

    if (!cmd_list || !cmd_table_buffer || !g_ahci_fis_buffer) return;

    for(int k=0; k<32; ++k) {
        cmd_list[k].ctba = (uint64_t)(uintptr_t)(cmd_table_buffer + (k * 256));
    }

    uint32_t ports_implemented = *(volatile uint32_t*)(uintptr_t)(ahci_base + 0x0C);

    // Auto-select a port. Two passes so a SATA disk on port 1 is preferred
    // over an ATAPI CD-ROM on port 0 (which is exactly the QEMU layout in
    // compile.md — `bus=ahci.0` for the CD, `bus=ahci.1` for the HDD).
    //
    // PxSIG (port offset 0x24) tells us what kind of device is attached:
    //   0x00000101 = SATA disk
    //   0xEB140101 = SATAPI / ATAPI (CD-ROM, DVD, etc.)
    //   0xC33C0101 = enclosure-management bridge
    //   0x96690101 = port multiplier
    //
    // Pass 1: claim the first SATA disk.
    // Pass 2: fall back to anything else (so we don't strand a CD-only
    // configuration with no port selected at all).
    auto try_select = [&](bool sata_only) -> bool {
        for (int i = 0; i < 32; i++) {
            if (!(ports_implemented & (1u << i))) continue;
            volatile HBA_PORT* p = (volatile HBA_PORT*)(uintptr_t)
                                    (ahci_base + 0x100 + (i * 0x80));
            uint32_t sig = p->sig;
            if (sata_only && sig != 0x00000101u) continue;
            if (ahci_port_setup(i)) {
                g_ahci_port = i;
                char msg[80];
                const char* kind = (sig == 0x00000101u) ? "SATA disk"
                                 : (sig == 0xEB140101u) ? "ATAPI (CD/DVD)"
                                 : "unknown";
                snprintf(msg, sizeof(msg),
                         "AHCI: port %d active (%s).\n", i, kind);
                wm.print_to_focused(msg);
                return true;
            }
        }
        return false;
    };
    if (try_select(true))  return;     // first SATA disk
    if (try_select(false)) return;     // fall back to anything
    wm.print_to_focused("AHCI: controller found but no active drive on any port.\n");
}bool fat32_init() {
    if (!ahci_base) return false;

    // Boot sector is always plaintext — read it raw regardless of crypto state
    bool was_enabled = g_fs_encryption_enabled;
    g_fs_encryption_enabled = false;

    char* buffer = new char[SECTOR_SIZE];
    if (!buffer) {
        g_fs_encryption_enabled = was_enabled;
        return false;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Sector 0 can be one of three things on a real disk:
    //   (a) Raw FAT32 boot sector — what mkfat32.py produces for QEMU/VMware.
    //       Identifiable by "FAT32   " string at offset 82.
    //   (b) Classic MBR — first 446 bytes are bootstrap, then a 4-entry
    //       partition table at offset 446, then signature 0x55 0xAA at 510.
    //       FAT32 partition types are 0x0B, 0x0C, 0x1B, 0x1C.
    //   (c) GPT protective MBR — signature 0x55 0xAA at 510, but the
    //       partition table contains exactly one entry of type 0xEE
    //       spanning the disk; the real GPT header lives at LBA 1, with
    //       128-byte entries starting at LBA 2 (or wherever the header
    //       says).
    //
    // We resolve the FAT32 partition's start LBA into partition_lba.
    // fat_start_sector / data_start_sector then carry absolute disk LBAs,
    // so cluster_to_lba and read_fat_entry keep working unchanged.
    // ─────────────────────────────────────────────────────────────────────
    if (read_write_sectors(g_ahci_port, 0, 1, false, buffer) != 0) {
        g_fs_encryption_enabled = was_enabled;
        delete[] buffer;
        return false;
    }

    uint64_t partition_lba = 0;
    bool     found_fat32   = false;
    bool     sector0_is_fat32 = (strncmp(buffer + 82, "FAT32", 5) == 0);
    bool     has_mbr_sig =
        ((uint8_t)buffer[510] == 0x55 && (uint8_t)buffer[511] == 0xAA);

    if (sector0_is_fat32) {
        // (a) Raw image. Sector 0 itself is the BPB. partition_lba stays 0.
        found_fat32 = true;
    } else if (has_mbr_sig) {
        // Detect GPT protective MBR vs classic MBR.
        bool is_protective_mbr = false;
        for (int i = 0; i < 4; i++) {
            if ((uint8_t)buffer[446 + i*16 + 4] == 0xEE) {
                is_protective_mbr = true;
                break;
            }
        }

        if (is_protective_mbr) {
            // ───── (c) GPT path ────────────────────────────────────────
            // Read GPT header at LBA 1. Header layout (only the fields
            // we need): signature "EFI PART" at offset 0, partition-
            // entry LBA at offset 72 (8 bytes), num_entries at offset 80
            // (4 bytes), entry_size at offset 84 (4 bytes).
            char* gpt_hdr = new char[SECTOR_SIZE];
            if (gpt_hdr &&
                read_write_sectors(g_ahci_port, 1, 1, false, gpt_hdr) == 0 &&
                strncmp(gpt_hdr, "EFI PART", 8) == 0)
            {
                uint64_t pe_lba   = *(uint64_t*)(gpt_hdr + 72);
                uint32_t pe_count = *(uint32_t*)(gpt_hdr + 80);
                uint32_t pe_size  = *(uint32_t*)(gpt_hdr + 84);

                // Sanity-cap; GPT spec mandates >= 128 entries, 128-byte size.
                if (pe_count > 256) pe_count = 256;
                if (pe_size  < 128 || pe_size > SECTOR_SIZE) pe_size = 128;

                uint32_t entries_per_sector = SECTOR_SIZE / pe_size;
                uint32_t sectors_to_read    =
                    (pe_count + entries_per_sector - 1) / entries_per_sector;

                char* entries = new char[SECTOR_SIZE];
                for (uint32_t s = 0;
                     entries && s < sectors_to_read && !found_fat32; s++)
                {
                    if (read_write_sectors(g_ahci_port, pe_lba + s, 1,
                                           false, entries) != 0) break;
                    for (uint32_t e = 0;
                         e < entries_per_sector && !found_fat32; e++)
                    {
                        char*    ent       = entries + e * pe_size;
                        uint64_t first_lba = *(uint64_t*)(ent + 32);
                        if (first_lba == 0) continue;   // empty slot

                        // Don't filter on type-GUID — just probe each
                        // partition's first sector for the FAT32 string.
                        // Avoids hard-coding the Microsoft Basic Data
                        // GUID (which most FAT32 ESP/data partitions
                        // use, but some installers vary).
                        if (read_write_sectors(g_ahci_port, first_lba, 1,
                                               false, buffer) != 0) continue;
                        if (strncmp(buffer + 82, "FAT32", 5) == 0) {
                            partition_lba = first_lba;
                            found_fat32   = true;
                        }
                    }
                }
                delete[] entries;
            }
            delete[] gpt_hdr;
        } else {
            // ───── (b) Classic MBR path ────────────────────────────────
            // Walk the 4-entry partition table. Take the first FAT32
            // partition whose boot sector verifies.
            for (int i = 0; i < 4 && !found_fat32; i++) {
                uint8_t* part = (uint8_t*)(buffer + 446 + i * 16);
                uint8_t  type = part[4];
                if (type != 0x0B && type != 0x0C &&
                    type != 0x1B && type != 0x1C) continue;

                uint64_t lba = (uint64_t) part[8]         |
                               ((uint64_t)part[9]  <<  8) |
                               ((uint64_t)part[10] << 16) |
                               ((uint64_t)part[11] << 24);
                if (lba == 0) continue;

                if (read_write_sectors(g_ahci_port, lba, 1, false, buffer) == 0
                    && strncmp(buffer + 82, "FAT32", 5) == 0)
                {
                    partition_lba = lba;
                    found_fat32   = true;
                }
            }
        }
    }

    g_fs_encryption_enabled = was_enabled;

    if (!found_fat32) {
        delete[] buffer;
        current_directory_cluster = 0;
        if (sector0_is_fat32) {
            // Shouldn't happen — we already set found_fat32 above.
        } else if (has_mbr_sig) {
            wm.print_to_focused("FAT32: partition table present, but no FAT32 partition.\n");
            wm.print_to_focused("  Create one (MBR type 0x0C, or GPT 'Microsoft Basic Data').\n");
        } else {
            wm.print_to_focused("FAT32: disk has no partition table and no FAT32 BPB.\n");
            wm.print_to_focused("  Either format the disk or write a raw FAT32 image.\n");
        }
        return false;
    }

    // `buffer` now holds the FAT32 BPB (sector 0 directly, or the
    // partition's first sector via MBR/GPT lookup).
    memcpy(&bpb, buffer, sizeof(bpb));
    delete[] buffer;

    if (strncmp(bpb.fil_sys_type, "FAT32", 5) != 0) {
        // Defensive: shouldn't trigger because we verified above.
        current_directory_cluster = 0;
        return false;
    }

    // fat_start_sector / data_start_sector hold ABSOLUTE disk LBAs.
    // partition_lba == 0 for the raw-image case, so QEMU/VMware
    // behaviour is byte-for-byte unchanged.
    g_partition_lba   = partition_lba;
    fat_start_sector  = partition_lba + bpb.rsvd_sec_cnt;
    data_start_sector = fat_start_sector + (bpb.num_fats * bpb.fat_sz32);
    current_directory_cluster = bpb.root_clus;

    if (partition_lba) {
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "FAT32: partition @ LBA %u, root_clus %u\n",
                 (unsigned)partition_lba, (unsigned)current_directory_cluster);
        wm.print_to_focused(msg);
    }
    return true;
}
uint64_t cluster_to_lba(uint32_t cluster) {
  return (uint64_t)(cluster - 2) * bpb.sec_per_clus + data_start_sector;
}

// Number of clusters in the FAT32 filesystem, computed entirely from BPB
// fields so it is independent of where on the disk the partition lives.
// The old formula `bpb.tot_sec32 - data_start_sector` worked only when
// data_start_sector was partition-relative; once we support MBR/GPT,
// data_start_sector holds the absolute disk LBA and the subtraction
// underflows. This helper sidesteps the issue.
static inline uint32_t fat32_data_sectors() {
    uint32_t reserved = bpb.rsvd_sec_cnt;
    uint32_t fats     = bpb.num_fats * bpb.fat_sz32;
    uint32_t fs_total = bpb.tot_sec32;
    if (fs_total <= reserved + fats) return 0;
    return fs_total - reserved - fats;
}
static inline uint32_t fat32_max_clusters() {
    if (bpb.sec_per_clus == 0) return 0;
    return fat32_data_sectors() / bpb.sec_per_clus + 2;
}void to_83_format(const char* filename, char* out) { memset(out, ' ', 11); int i = 0, j = 0; while (filename[i] && filename[i] != '.' && j < 8) { out[j++] = (filename[i] >= 'a' && filename[i] <= 'z') ? (filename[i]-32) : filename[i]; i++; } if(filename[i] == '.') i++; j=8; while(filename[i] && j<11) { out[j++] = (filename[i] >= 'a' && filename[i] <= 'z') ? (filename[i]-32) : filename[i]; i++; } }

void from_83_format(const char* fat_name, char* out) {
    int i, j = 0;
    // Process the name part (before the extension)
    for (i = 0; i < 8 && fat_name[i] != ' '; i++) {
        // Only convert uppercase letters to lowercase
        out[j++] = (fat_name[i] >= 'A' && fat_name[i] <= 'Z') ? fat_name[i] + 32 : fat_name[i];
    }
    
    // Process the extension part, if it exists
    if (fat_name[8] != ' ') {
        out[j++] = '.';
        for (i = 8; i < 11 && fat_name[i] != ' '; i++) {
            // Only convert uppercase letters to lowercase
            out[j++] = (fat_name[i] >= 'A' && fat_name[i] <= 'Z') ? fat_name[i] + 32 : fat_name[i];
        }
    }
    out[j] = '\0';
}

void fat32_get_fne_from_entry(fat_dir_entry_t* entry, char* out) {
    from_83_format(entry->name, out);
}

uint32_t read_fat_entry(uint32_t cluster) {
    uint8_t* fat_sector = new uint8_t[SECTOR_SIZE];
    uint32_t fat_offset = cluster * 4;

    read_write_sectors(g_ahci_port, fat_start_sector + (fat_offset / SECTOR_SIZE), 1, false, fat_sector);
    uint32_t value = *(uint32_t*)(fat_sector + (fat_offset % SECTOR_SIZE)) & 0x0FFFFFFF;
    delete[] fat_sector;
    return value;
}

bool write_fat_entry(uint32_t cluster, uint32_t value) {
    uint8_t* fat_sector = new uint8_t[SECTOR_SIZE];
    uint32_t fat_offset = cluster * 4;
    uint32_t fat_sector_index = fat_offset / SECTOR_SIZE;
    uint32_t sector_num = fat_start_sector + fat_sector_index;

    read_write_sectors(g_ahci_port, sector_num, 1, false, fat_sector);
    *(uint32_t*)(fat_sector + (fat_offset % SECTOR_SIZE)) =
        (*(uint32_t*)(fat_sector + (fat_offset % SECTOR_SIZE)) & 0xF0000000) |
        (value & 0x0FFFFFFF);

    bool success = read_write_sectors(g_ahci_port, sector_num, 1, true, fat_sector) == 0;

    // ─────────────────────────────────────────────────────────────────────
    // Mirror to FAT2. The FAT32 spec says: if BPB_ExtFlags bit 7 is clear
    // (the default), the FAT is mirrored — every update must hit FAT1 AND
    // FAT2. Windows CHKDSK and Linux dosfsck both treat FAT1/FAT2 mismatch
    // as filesystem corruption and may "repair" by overwriting the live
    // FAT from the stale one. The old code wrote only FAT1, guaranteeing
    // every file created by this OS looked corrupted to any other OS.
    //
    // If ext_flags bit 7 is set, only the FAT specified by bits 0-3 is
    // active; honour that and skip the mirror.
    // ─────────────────────────────────────────────────────────────────────
    bool mirror_fats = (bpb.ext_flags & 0x0080) == 0;
    if (success && mirror_fats && bpb.num_fats >= 2) {
        uint32_t fat2_sector_num = fat_start_sector + bpb.fat_sz32 + fat_sector_index;
        success = read_write_sectors(g_ahci_port, fat2_sector_num, 1, true, fat_sector) == 0;
    }

    delete[] fat_sector;
    return success;
}

uint32_t find_free_cluster() {
    uint32_t max_clusters = fat32_max_clusters();
    for (uint32_t i = 2; i < max_clusters; i++) if (read_fat_entry(i) == FAT_FREE_CLUSTER) return i;
    return 0;
}

uint32_t allocate_cluster() {
    uint32_t free_cluster = find_free_cluster();
    if (free_cluster != 0) write_fat_entry(free_cluster, FAT_END_OF_CHAIN);
    return free_cluster;
}

void free_cluster_chain(uint32_t start_cluster) {
    uint32_t current = start_cluster;
    while(current < FAT_END_OF_CHAIN) { uint32_t next = read_fat_entry(current); write_fat_entry(current, FAT_FREE_CLUSTER); current = next; }
}

// Hinted free-cluster scan: starts from `start_from` instead of cluster 2.
// allocate_cluster_chain uses this to avoid the O(N^2) cost of the original
// "always scan from cluster 2" loop — that read the same FAT sectors over
// and over (millions of redundant reads for a 1 MB file at 512-byte
// clusters). Falls back to a full scan if nothing was found above the hint.
static uint32_t find_free_cluster_hinted(uint32_t start_from) {
    uint32_t max_clusters = fat32_max_clusters();
    if (start_from < 2) start_from = 2;
    for (uint32_t i = start_from; i < max_clusters; i++)
        if (read_fat_entry(i) == FAT_FREE_CLUSTER) return i;
    for (uint32_t i = 2; i < start_from && i < max_clusters; i++)
        if (read_fat_entry(i) == FAT_FREE_CLUSTER) return i;
    return 0;
}

uint32_t allocate_cluster_chain(uint32_t num_clusters) {
    if(num_clusters == 0) return 0;
    // Find the first free cluster (full scan, once).
    uint32_t first = allocate_cluster();
    if(first == 0) return 0;
    uint32_t current = first;
    // For the rest of the chain, scan FORWARD from the last cluster we
    // grabbed — on a freshly-formatted disk the next free cluster is
    // almost always current+1, which costs one FAT read per allocation
    // instead of (number-allocated-so-far) FAT reads.
    for(uint32_t i = 1; i < num_clusters; i++) {
        uint32_t next = find_free_cluster_hinted(current + 1);
        if(next == 0) { free_cluster_chain(first); return 0; }
        write_fat_entry(next,    FAT_END_OF_CHAIN); // mark allocated
        write_fat_entry(current, next);             // link previous → next
        current = next;
    }
    return first;
}

bool read_data_from_clusters(uint32_t start_cluster, void* data, uint32_t size) {
    if (size == 0) return true;
    uint8_t* data_ptr = (uint8_t*)data;
    uint32_t remaining = size;
    uint32_t current_cluster = start_cluster;
    uint32_t cluster_size = bpb.sec_per_clus * SECTOR_SIZE;

    while (current_cluster >= 2 && current_cluster < FAT_END_OF_CHAIN && remaining > 0) {
        uint32_t to_read = (remaining > cluster_size) ? cluster_size : remaining;
        uint8_t* cluster_buf = new uint8_t[cluster_size];
        memset(cluster_buf, 0, cluster_size); // Clear buffer
        if(read_write_sectors(g_ahci_port, cluster_to_lba(current_cluster), bpb.sec_per_clus, false, cluster_buf) != 0) { 
            delete[] cluster_buf; 
            return false; 
        }
        memcpy(data_ptr, cluster_buf, to_read);
        delete[] cluster_buf;
        data_ptr += to_read;
        remaining -= to_read;
        if (remaining > 0) current_cluster = read_fat_entry(current_cluster);
        else break;
    }
    return true;
}

bool write_data_to_clusters(uint32_t start_cluster, const void* data, uint32_t size) {
    if (size == 0) return true;
    const uint8_t* data_ptr = (const uint8_t*)data;
    uint32_t remaining = size;
    uint32_t current_cluster = start_cluster;
    uint32_t cluster_size = bpb.sec_per_clus * SECTOR_SIZE;
    uint8_t* cluster_buf = new uint8_t[cluster_size];

    while (current_cluster >= 2 && current_cluster < FAT_END_OF_CHAIN && remaining > 0) {
        uint32_t to_write = (remaining > cluster_size) ? cluster_size : remaining;
        memset(cluster_buf, 0, cluster_size);
        memcpy(cluster_buf, data_ptr, to_write);
        if (read_write_sectors(g_ahci_port, cluster_to_lba(current_cluster), bpb.sec_per_clus, true, cluster_buf) != 0) { 
            delete[] cluster_buf; 
            return false; 
        }
        data_ptr += to_write;
        remaining -= to_write;
        if (remaining > 0) current_cluster = read_fat_entry(current_cluster);
        else break;
    }
    delete[] cluster_buf;
    return true;
}

uint32_t clusters_needed(uint32_t size) {
    if (bpb.sec_per_clus == 0) return 0;
    uint32_t cluster_size = bpb.sec_per_clus * SECTOR_SIZE;
    return (size + cluster_size - 1) / cluster_size;
}

void fat32_list_files() {
    if (!ahci_base || !current_directory_cluster) {
        wm.print_to_focused("Filesystem not ready.\n");
        return;
    }
    uint32_t cluster_bytes = bpb.sec_per_clus * SECTOR_SIZE;
    uint8_t* buffer = new uint8_t[cluster_bytes];

    wm.print_to_focused("Name                           Size\n");
    char lfn_buf[256] = {0};

    // FAT32 directories are ordinary cluster chains, not single clusters.
    // Once enough entries exist to fill one cluster (trivially easy: a
    // 512-byte cluster only holds 16 entries), the directory spills into
    // a second, third, ... cluster via the FAT chain. The old code only
    // ever read the FIRST cluster, so any files sitting in later clusters
    // — very common right after copying a batch of files in from another
    // OS — silently vanished from `ls`. We now walk the whole chain and
    // only stop when we hit a genuine end-of-directory marker (name[0] ==
    // 0x00), exactly like real FAT32 implementations do.
    uint32_t cluster = current_directory_cluster;
    bool end_of_dir = false;
    while (!end_of_dir && cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        if (read_write_sectors(g_ahci_port, cluster_to_lba(cluster), bpb.sec_per_clus, false, buffer) != 0) {
            wm.print_to_focused("Read error\n");
            break;
        }

        for (uint32_t i = 0; i < cluster_bytes; i += sizeof(fat_dir_entry_t)) {
            fat_dir_entry_t* entry = (fat_dir_entry_t*)(buffer + i);

            if (entry->name[0] == 0x00) { end_of_dir = true; break; }
            if ((uint8_t)entry->name[0] == DELETED_ENTRY) {
                lfn_buf[0] = '\0';
                continue;
            }
            if (entry->name[0] == '.') continue;

            if (entry->attr == ATTR_LONG_NAME) {
                fat_lfn_entry_t* lfn = (fat_lfn_entry_t*)entry;
                if (lfn->order & 0x40) lfn_buf[0] = '\0';

                char name_part[14] = {0};
                int k = 0;
                auto extract = [&](uint16_t val) {
                    if (k < 13 && val != 0x0000 && val != 0xFFFF) name_part[k++] = (char)val;
                };
                for(int j=0; j<5; j++) extract(lfn->name1[j]);
                for(int j=0; j<6; j++) extract(lfn->name2[j]);
                for(int j=0; j<2; j++) extract(lfn->name3[j]);

                memmove(lfn_buf + k, lfn_buf, strlen(lfn_buf) + 1);
                memcpy(lfn_buf, name_part, k);

            } else if (!(entry->attr & ATTR_VOLUME_ID)) {
                char line[120];
                char fname_83[13];
                const char* name_to_print;

                if (lfn_buf[0] != '\0') {
                    name_to_print = lfn_buf;
                } else {
                    from_83_format(entry->name, fname_83);
                    name_to_print = fname_83;
                }

                // Manually copy and pad the filename to 30 characters
                int name_len = strlen(name_to_print);
                int copy_len = (name_len > 30) ? 30 : name_len;
                memcpy(line, name_to_print, copy_len);
                for (int k = copy_len; k < 30; ++k) {
                    line[k] = ' ';
                }
                line[30] = '\0'; // Terminate after the padded name

                // Use a simple snprintf for just the size
                snprintf(line + 30, 90, " %d\n", entry->file_size);

                wm.print_to_focused(line);
                lfn_buf[0] = '\0'; // Reset for next entry
            }
        }

        if (!end_of_dir) cluster = read_fat_entry(cluster);
    }
    delete[] buffer;
}
int fat32_write_file(const char* filename, const void* data, uint32_t size) {
    // First, safely remove the file if it already exists to handle overwrites correctly.
    fat32_remove_file(filename);

    char target_83[11];
    to_83_format(filename, target_83);
    uint32_t first_cluster = 0;

    if (size > 0) {
        uint32_t num_clusters = clusters_needed(size);
        if (num_clusters == 0) return -1;
        
        first_cluster = allocate_cluster_chain(num_clusters);
        if (first_cluster == 0) return -1; // Out of space
        if (!write_data_to_clusters(first_cluster, data, size)) {
            free_cluster_chain(first_cluster);
            return -1; // Write error
        }
    }

    uint8_t* dir_buf = new uint8_t[SECTOR_SIZE];

    // Walk the directory's ENTIRE cluster chain looking for a free slot,
    // instead of only ever looking at the first cluster. A directory is
    // just a cluster chain like any file; a 512-byte cluster only holds
    // 16 entries, so it's trivial to fill the first cluster and spill
    // into a second one — something any real OS handles transparently.
    // Before this fix, once cluster #1 was full this function returned
    // "Directory is full" even with the whole rest of the disk empty,
    // and any files that DID make it into a later cluster (e.g. written
    // by another OS) were invisible to fat32_find_entry/list_files too.
    uint32_t cluster = current_directory_cluster;
    uint32_t last_cluster = cluster;
    while (cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        last_cluster = cluster;
        for (uint8_t s = 0; s < bpb.sec_per_clus; s++) {
            uint64_t sector_lba = cluster_to_lba(cluster) + s;
            if (read_write_sectors(g_ahci_port, sector_lba, 1, false, dir_buf) != 0) continue;

            for (uint16_t e = 0; e < SECTOR_SIZE / sizeof(fat_dir_entry_t); e++) {
                fat_dir_entry_t* entry = (fat_dir_entry_t*)(dir_buf + e * sizeof(fat_dir_entry_t));
                if (entry->name[0] == 0x00 || (uint8_t)entry->name[0] == DELETED_ENTRY) {
                    // Found a free slot, create the entry.
                    memset(entry, 0, sizeof(fat_dir_entry_t));
                    memcpy(entry->name, target_83, 11);
                    entry->attr = ATTR_ARCHIVE;
                    entry->file_size = size;
                    entry->fst_clus_lo = first_cluster & 0xFFFF;
                    entry->fst_clus_hi = (first_cluster >> 16) & 0xFFFF;

                    if (read_write_sectors(g_ahci_port, sector_lba, 1, true, dir_buf) == 0) {
                        delete[] dir_buf;
                        return 0; // Success
                    } else {
                        delete[] dir_buf;
                        if(first_cluster > 0) free_cluster_chain(first_cluster);
                        return -1; // Directory write error
                    }
                }
            }
        }
        cluster = read_fat_entry(cluster);
    }

    // Every existing directory cluster is completely full (no 0x00 and no
    // deleted-entry slot anywhere in the chain): grow the directory by
    // appending a fresh cluster, exactly like a real FAT32 driver would.
    uint32_t new_dir_cluster = allocate_cluster();
    if (new_dir_cluster == 0) {
        delete[] dir_buf;
        if (first_cluster > 0) free_cluster_chain(first_cluster);
        return -1; // Disk is genuinely full, can't grow the directory
    }

    uint32_t cluster_bytes = bpb.sec_per_clus * SECTOR_SIZE;
    uint8_t* new_clus_buf = new uint8_t[cluster_bytes];
    memset(new_clus_buf, 0, cluster_bytes); // zeroed => first unused entry marks end-of-directory

    fat_dir_entry_t* new_entry = (fat_dir_entry_t*)new_clus_buf;
    memcpy(new_entry->name, target_83, 11);
    new_entry->attr = ATTR_ARCHIVE;
    new_entry->file_size = size;
    new_entry->fst_clus_lo = first_cluster & 0xFFFF;
    new_entry->fst_clus_hi = (first_cluster >> 16) & 0xFFFF;

    bool wrote_ok = (read_write_sectors(g_ahci_port, cluster_to_lba(new_dir_cluster), bpb.sec_per_clus, true, new_clus_buf) == 0);
    delete[] new_clus_buf;
    delete[] dir_buf;

    if (!wrote_ok) {
        free_cluster_chain(new_dir_cluster);
        if (first_cluster > 0) free_cluster_chain(first_cluster);
        return -1;
    }

    // Link the new cluster onto the end of the directory's FAT chain.
    write_fat_entry(last_cluster, new_dir_cluster);
    write_fat_entry(new_dir_cluster, FAT_END_OF_CHAIN);
    return 0;
}

char* fat32_read_file_as_string(const char* filename) {
    char target[11]; to_83_format(filename, target);
    uint8_t* dir_buf = new uint8_t[SECTOR_SIZE];
    // Walk the full directory cluster chain (see fat32_list_files() for
    // why this matters) instead of stopping after the first cluster.
    uint32_t cluster = current_directory_cluster;
    while (cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        for (uint8_t s = 0; s < bpb.sec_per_clus; s++) {
            if (read_write_sectors(g_ahci_port, cluster_to_lba(cluster) + s, 1, false, dir_buf) != 0) { delete[] dir_buf; return nullptr; }
            for (uint16_t e = 0; e < SECTOR_SIZE / sizeof(fat_dir_entry_t); e++) {
                fat_dir_entry_t* entry = (fat_dir_entry_t*)(dir_buf + e * sizeof(fat_dir_entry_t));
                if (entry->name[0] == 0x00) { delete[] dir_buf; return nullptr; }
                if (memcmp(entry->name, target, 11) == 0) {
                    uint32_t size = entry->file_size;
                    if(size == 0) { delete[] dir_buf; char* empty = new char[1]; empty[0] = '\0'; return empty; }
                    char* data = new char[size + 1];
                    if (read_data_from_clusters((entry->fst_clus_hi << 16) | entry->fst_clus_lo, data, size)) {
                        data[size] = '\0';
                        delete[] dir_buf;
                        return data;
                    }
                    delete[] data; delete[] dir_buf; return nullptr;
                }
            }
        }
        cluster = read_fat_entry(cluster);
    }
    delete[] dir_buf; return nullptr;
}

int fat32_find_entry(const char* filename, fat_dir_entry_t* entry_out, uint32_t* sector_out, uint32_t* offset_out) {
    char lfn_buf[256] = {0};
    uint8_t current_checksum = 0;

    // Walk the directory's FULL cluster chain, not just its first cluster.
    // See the comment in fat32_list_files() for why this matters: any
    // entry copied in from another OS that landed past cluster #1 used to
    // be completely invisible to this lookup, which made "cp"/"cat"/open
    // silently fail on files that clearly existed on disk.
    uint8_t* dir_buf = new uint8_t[SECTOR_SIZE];
    uint32_t cluster = current_directory_cluster;
    while (cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        for(uint8_t s=0; s<bpb.sec_per_clus; ++s) {
            uint32_t current_sector = cluster_to_lba(cluster) + s;
            if(read_write_sectors(g_ahci_port, current_sector, 1, false, dir_buf) != 0) {
                delete[] dir_buf;
                return -1;
            }

            for(uint16_t e=0; e < SECTOR_SIZE / sizeof(fat_dir_entry_t); ++e) {
                fat_dir_entry_t* entry = (fat_dir_entry_t*)(dir_buf + e*sizeof(fat_dir_entry_t));
                if(entry->name[0] == 0x00) { delete[] dir_buf; return -1; }
                if((uint8_t)entry->name[0] == DELETED_ENTRY) { lfn_buf[0] = '\0'; continue; }

                if(entry->attr == ATTR_LONG_NAME) {
                    fat_lfn_entry_t* lfn = (fat_lfn_entry_t*)entry;
                    if (lfn->order & 0x40) {
                        lfn_buf[0] = '\0';
                        current_checksum = lfn->checksum;
                    }

                    char name_part[14] = {0};
                    int k = 0;
                    auto extract = [&](uint16_t val) {
                        if (k < 13 && val != 0x0000 && val != 0xFFFF) name_part[k++] = (char)val;
                    };
                    for(int j=0; j<5; j++) extract(lfn->name1[j]);
                    for(int j=0; j<6; j++) extract(lfn->name2[j]);
                    for(int j=0; j<2; j++) extract(lfn->name3[j]);

                    memmove(lfn_buf + k, lfn_buf, strlen(lfn_buf) + 1);
                    memcpy(lfn_buf, name_part, k);

                } else if (!(entry->attr & ATTR_VOLUME_ID)) {
                    bool match = false;
                    if(lfn_buf[0] != '\0' && lfn_checksum((unsigned char*)entry->name) == current_checksum) {
                        if(strcmp(lfn_buf, filename) == 0) match = true;
                    } else {
                        char sfn_name[13];
                        from_83_format(entry->name, sfn_name);
                        if(strcmp(sfn_name, filename) == 0) match = true;
                    }

                    lfn_buf[0] = '\0';

                    if(match) {
                        memcpy(entry_out, entry, sizeof(fat_dir_entry_t));
                        *sector_out = current_sector;
                        *offset_out = e * sizeof(fat_dir_entry_t);
                        delete[] dir_buf;
                        return 0;
                    }
                }
            }
        }
        cluster = read_fat_entry(cluster);
    }
    delete[] dir_buf;
    return -1;
}
// Guest-disk-wrapper helper (see bochs_glue.cpp's bochs_guest_disk_cmd):
// get a file's size without reading its contents. Used both for the
// guest's STAT command and internally by READ, so a too-small buffer
// fails fast (with the real size reported back) instead of paying for
// a full fat32_read_file_as_string() alloc+copy first.
int fat32_stat_file(const char* filename, uint32_t* size_out) {
    fat_dir_entry_t entry;
    uint32_t sector, offset;
    if (fat32_find_entry(filename, &entry, &sector, &offset) != 0) return -1;
    if (size_out) *size_out = entry.file_size;
    return 0;
}

int fat32_list_directory(const char* path, fat_dir_entry_t* buffer, int max_entries) {
    // This implementation ignores 'path' and lists the current directory for simplicity.
    if (!ahci_base || !current_directory_cluster || !buffer) {
        return 0;
    }

    uint32_t cluster_bytes = bpb.sec_per_clus * SECTOR_SIZE;
    uint8_t* dir_sector_buf = new uint8_t[cluster_bytes];

    int count = 0;
    // Walk the full directory cluster chain (see fat32_list_files() for
    // why: entries past the first cluster used to be invisible here too,
    // which is why file explorer / desktop icon lists silently dropped
    // files copied in from another OS).
    uint32_t cluster = current_directory_cluster;
    bool end_of_dir = false;
    while (!end_of_dir && count < max_entries && cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        if (read_write_sectors(g_ahci_port, cluster_to_lba(cluster), bpb.sec_per_clus, false, dir_sector_buf) != 0) {
            break; // Read error
        }

        for (uint32_t i = 0; i < cluster_bytes; i += sizeof(fat_dir_entry_t)) {
            if (count >= max_entries) break;

            fat_dir_entry_t* entry = (fat_dir_entry_t*)(dir_sector_buf + i);

            if (entry->name[0] == 0x00) { end_of_dir = true; break; } // End of directory
            if ((uint8_t)entry->name[0] == DELETED_ENTRY) continue; // Skip deleted entries
            if (entry->attr == ATTR_LONG_NAME || (entry->attr & ATTR_VOLUME_ID)) continue; // Skip LFN and Volume ID

            // This is a valid file or directory, so copy it to the output buffer
            memcpy(&buffer[count], entry, sizeof(fat_dir_entry_t));
            count++;
        }

        if (!end_of_dir) cluster = read_fat_entry(cluster);
    }

    delete[] dir_sector_buf;
    return count;
}
int fat32_remove_file(const char* filename) {
    fat_dir_entry_t entry;
    uint32_t sector, offset;
    if(fat32_find_entry(filename, &entry, &sector, &offset) != 0) return -1;
    uint32_t start_cluster = (entry.fst_clus_hi << 16) | entry.fst_clus_lo;
    if(start_cluster != 0) free_cluster_chain(start_cluster);
    
    uint8_t* dir_buf = new uint8_t[SECTOR_SIZE];
    read_write_sectors(g_ahci_port, sector, 1, false, dir_buf);
    ((fat_dir_entry_t*)(dir_buf + offset))->name[0] = DELETED_ENTRY;
    read_write_sectors(g_ahci_port, sector, 1, true, dir_buf);
    delete[] dir_buf;
    return 0;
}
// ADD THIS NEW FUNCTION after fat32_rename_file
int fat32_copy_file(const char* src_path, const char* dest_path) {
    fat_dir_entry_t entry;
    uint32_t sector, offset;

    // 1. Find the source file and get its info
    if (fat32_find_entry(src_path, &entry, &sector, &offset) != 0) {
        return -1; // Source file not found
    }

    if (entry.file_size == 0) {
        // Handle zero-byte files
        return fat32_write_file(dest_path, nullptr, 0);
    }
    
    // 2. Allocate memory and read the source file's content
    uint8_t* content_buffer = new uint8_t[entry.file_size];
    if (!content_buffer) {
        return -2; // Out of memory
    }

    uint32_t start_cluster = (entry.fst_clus_hi << 16) | entry.fst_clus_lo;
    if (!read_data_from_clusters(start_cluster, content_buffer, entry.file_size)) {
        delete[] content_buffer;
        return -3; // Failed to read source file
    }

    // 3. Write the content to the destination file
    int result = fat32_write_file(dest_path, content_buffer, entry.file_size);
    
    delete[] content_buffer;
    return (result == 0) ? 0 : -4; // Return 0 on success, else write error
}
int fat32_rename_file(const char* old_name, const char* new_name) {
    fat_dir_entry_t entry;
    uint32_t sector, offset;
    fat_dir_entry_t dummy_entry;
    uint32_t dummy_sector, dummy_offset;

    // 1. Check if new_name already exists. If so, fail.
    if (fat32_find_entry(new_name, &dummy_entry, &dummy_sector, &dummy_offset) == 0) {
        return -1; // Destination file already exists
    }

    // 2. Find the old file. If it doesn't exist, fail.
    if (fat32_find_entry(old_name, &entry, &sector, &offset) != 0) {
        return -1; // Source file not found
    }
    
    // 3. Read, modify, and write back the directory sector.
    uint8_t* dir_buf = new uint8_t[SECTOR_SIZE];
    if (read_write_sectors(g_ahci_port, sector, 1, false, dir_buf) != 0) {
        delete[] dir_buf;
        return -1;
    }

    fat_dir_entry_t* target_entry = (fat_dir_entry_t*)(dir_buf + offset);
    to_83_format(new_name, target_entry->name);
    
    if (read_write_sectors(g_ahci_port, sector, 1, true, dir_buf) != 0) {
        delete[] dir_buf;
        return -1;
    }

    delete[] dir_buf;
    return 0; // Success
}
void fat32_format() {
    if(!ahci_base) {
        wm.print_to_focused("AHCI disk not found. Cannot format.\n");
        return;
    }
    wm.print_to_focused("WARNING: This is a destructive operation!\nFormatting disk...\n");

    // ─────────────────────────────────────────────────────────────────────
    // Decide partition layout upfront.
    //
    // On a raw disk (g_partition_lba == 0) we will write an MBR and place
    // the FAT32 partition at PART_START_LBA (2048 = 1 MiB, the standard
    // alignment used by fdisk/parted/mkfs.fat).  On a pre-partitioned disk
    // we format in-place at g_partition_lba.
    //
    // IMPORTANT: tot_sec32 in the BPB is the sector count OF THE PARTITION,
    // not the whole disk.  Windows fastfat validates:
    //   cluster_count = (tot_sec32 - rsvd_sec_cnt - num_fats*fat_sz32)
    //                   / sec_per_clus
    // and requires cluster_count >= 65525 for FAT32.  Using the whole-disk
    // sector count here inflates the number and causes chkdsk to report
    // "The volume size is too big" or refuse to mount on small images.
    // ─────────────────────────────────────────────────────────────────────
    const uint32_t PART_START_LBA = 2048;   // 1 MiB boundary (standard)
    const bool raw_disk = (g_partition_lba == 0);

    // Effective partition start that will end up in g_partition_lba after format.
    const uint64_t new_part_lba = raw_disk ? PART_START_LBA : g_partition_lba;

    // Total disk size in sectors (128 MB image).  The partition occupies
    // disk_total_sectors - new_part_lba sectors.
    const uint32_t disk_total_sectors = (128u * 1024u * 1024u) / 512u;
    // Guard: partition must fit within the disk.
    if (new_part_lba >= disk_total_sectors) {
        wm.print_to_focused("Error: partition start beyond disk end.\n");
        return;
    }
    const uint32_t part_total_sectors = disk_total_sectors - (uint32_t)new_part_lba;

    fat32_bpb_t new_bpb;
    memset(&new_bpb, 0, sizeof(fat32_bpb_t));
    new_bpb.jmp[0] = 0xEB; new_bpb.jmp[1] = 0x58; new_bpb.jmp[2] = 0x90;
    memcpy(new_bpb.oem, "MSWIN4.1", 8);
    new_bpb.bytes_per_sec = 512;
    new_bpb.rsvd_sec_cnt  = 32;
    new_bpb.num_fats      = 2;
    new_bpb.root_ent_cnt  = 0;      // must be 0 for FAT32
    new_bpb.tot_sec16     = 0;      // must be 0 for FAT32 (use tot_sec32)
    new_bpb.media         = 0xF8;
    new_bpb.fat_sz16      = 0;      // must be 0 for FAT32 (use fat_sz32)
    new_bpb.sec_per_trk   = 32;
    new_bpb.num_heads     = 64;
    new_bpb.hidd_sec      = (uint32_t)new_part_lba;  // sectors before this partition
    new_bpb.tot_sec32     = part_total_sectors;        // partition size, NOT disk size

    // ── sec_per_clus: smallest value that yields >= 65525 clusters ────────
    {
        uint8_t spc_candidates[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
        uint8_t chosen = 128; // safe fallback
        for (uint32_t k = 0; k < sizeof(spc_candidates); k++) {
            uint8_t spc = spc_candidates[k];
            uint32_t tmp1 = part_total_sectors - new_bpb.rsvd_sec_cnt;
            uint32_t tmp2 = (256u * spc + new_bpb.num_fats) / 2u;
            uint32_t fat_sz = (tmp1 + tmp2 - 1u) / tmp2;
            uint32_t data_sec = part_total_sectors
                                - new_bpb.rsvd_sec_cnt
                                - new_bpb.num_fats * fat_sz;
            uint32_t clusters = data_sec / spc;
            if (clusters >= 65525u) { chosen = spc; break; }
        }
        new_bpb.sec_per_clus = chosen;
    }

    // ── fat_sz32 from Microsoft FAT spec section 3.5 ─────────────────────
    {
        uint32_t tmp1 = part_total_sectors - new_bpb.rsvd_sec_cnt;
        uint32_t tmp2 = (256u * new_bpb.sec_per_clus + new_bpb.num_fats) / 2u;
        new_bpb.fat_sz32 = (tmp1 + tmp2 - 1u) / tmp2;
    }

    new_bpb.root_clus   = 2;
    new_bpb.fs_info     = 1;        // FSInfo at partition + 1
    new_bpb.bk_boot_sec = 6;        // backup boot sector at partition + 6
    new_bpb.ext_flags   = 0;        // mirror FATs
    new_bpb.fs_ver      = 0x0000;   // FAT32 version 0.0 — required by spec
    new_bpb.drv_num     = 0x80;
    new_bpb.res1        = 0;
    new_bpb.boot_sig    = 0x29;
    new_bpb.vol_id      = 0x12345678;
    memcpy(new_bpb.vol_lab,      "MYOS VOL   ", 11);
    memcpy(new_bpb.fil_sys_type, "FAT32   ",    8);

    // ── Commit globals NOW, before any sector writes that depend on them ──
    // All write helpers (write_fat_entry, cluster_to_lba, …) use the
    // module-level fat_start_sector / data_start_sector globals, so they
    // must be correct before we call them — not after.
    memcpy(&bpb, &new_bpb, sizeof(fat32_bpb_t));
    g_partition_lba   = new_part_lba;
    fat_start_sector  = (uint32_t)(new_part_lba + bpb.rsvd_sec_cnt);
    data_start_sector = fat_start_sector + (bpb.num_fats * bpb.fat_sz32);

    // ── Write MBR (raw disk only) ─────────────────────────────────────────
    if (raw_disk) {
        wm.print_to_focused("Writing MBR and partition table...\n");
        char* mbr = new char[SECTOR_SIZE];
        memset(mbr, 0, SECTOR_SIZE);

        // Minimal x86 bootstrap stub (prints "No bootable OS", halts).
        static const uint8_t boot_stub[] = {
            0xFA,                         // CLI
            0x31, 0xC0,                   // XOR AX, AX
            0x8E, 0xD0,                   // MOV SS, AX
            0xBC, 0x00, 0x7C,             // MOV SP, 0x7C00
            0xFB,                         // STI
            0x0E,                         // PUSH CS
            0x1F,                         // POP DS
            0xBE, 0x1E, 0x7C,             // MOV SI, msg_offset
            0xAC,                         // LODSB
            0x08, 0xC0,                   // OR AL, AL
            0x74, 0x09,                   // JZ halt
            0xB4, 0x0E,                   // MOV AH, 0x0E
            0xBB, 0x07, 0x00,             // MOV BX, 7
            0xCD, 0x10,                   // INT 0x10
            0xEB, 0xF2,                   // JMP loop
            0xF4,                         // HLT
            0xEB, 0xFD,                   // JMP halt
            'N','o',' ','b','o','o','t','a','b','l','e',' ','O','S','\r','\n', 0x00
        };
        memcpy(mbr, boot_stub, sizeof(boot_stub));

        // Partition table entry 0: type 0x0C (FAT32 LBA), bootable.
        // CHS values set to 0xFE/0xFF/0xFF (LBA-mode placeholder, per
        // the convention used by Windows Disk Management and fdisk).
        uint8_t* pt = (uint8_t*)mbr + 446;
        pt[0] = 0x80;                                // bootable
        pt[1] = 0xFE; pt[2] = 0xFF; pt[3] = 0xFF;  // CHS start placeholder
        pt[4] = 0x0C;                                // type: FAT32 LBA
        pt[5] = 0xFE; pt[6] = 0xFF; pt[7] = 0xFF;  // CHS end placeholder
        *(uint32_t*)(pt +  8) = PART_START_LBA;
        *(uint32_t*)(pt + 12) = part_total_sectors;
        // Entries 1–3: zeroed (unused).

        mbr[510] = (char)0x55;
        mbr[511] = (char)0xAA;

        bool was = g_fs_encryption_enabled;
        g_fs_encryption_enabled = false;
        if (read_write_sectors(g_ahci_port, 0, 1, true, mbr) != 0)
            wm.print_to_focused("Warning: MBR write failed.\n");
        else
            wm.print_to_focused("MBR + partition table written.\n");
        g_fs_encryption_enabled = was;
        delete[] mbr;
    }

    // ── Write VBR (Volume Boot Record = BPB sector) at new_part_lba ──────
    wm.print_to_focused("Writing volume boot record...\n");
    char* boot_sector_buffer = new char[SECTOR_SIZE];
    memset(boot_sector_buffer, 0, SECTOR_SIZE);
    memcpy(boot_sector_buffer, &new_bpb, sizeof(fat32_bpb_t));
    boot_sector_buffer[510] = (char)0x55;
    boot_sector_buffer[511] = (char)0xAA;

    bool boot_sec_was_enc = g_fs_encryption_enabled;
    g_fs_encryption_enabled = false;   // BPB/FSInfo must be plaintext

    if (read_write_sectors(g_ahci_port, new_part_lba, 1, true, boot_sector_buffer) != 0) {
        wm.print_to_focused("Error: Failed to write volume boot record.\n");
        delete[] boot_sector_buffer;
        g_fs_encryption_enabled = boot_sec_was_enc;
        return;
    }
    // Backup VBR at partition + bk_boot_sec (=6).
    wm.print_to_focused("Writing backup boot sector...\n");
    if (read_write_sectors(g_ahci_port, new_part_lba + new_bpb.bk_boot_sec, 1,
                           true, boot_sector_buffer) != 0)
        wm.print_to_focused("Warning: backup boot sector write failed.\n");

    // ── Write FSInfo sector at partition + fs_info (=1) ───────────────────
    // Signatures per FAT32 spec Table 9:
    //   offset   0  LeadSig  = 0x41615252  ("RRaA")
    //   offset 484  StrucSig = 0x61417272  ("rrAa")
    //   offset 488  FreeCount= 0xFFFFFFFF  (unknown — recompute on first mount)
    //   offset 492  NextFree = 0xFFFFFFFF  (unknown — search from cluster 2)
    //   offset 508  TrailSig = 0xAA550000
    wm.print_to_focused("Writing FSInfo sector...\n");
    memset(boot_sector_buffer, 0, SECTOR_SIZE);
    *(uint32_t*)(boot_sector_buffer +   0) = 0x41615252u;
    *(uint32_t*)(boot_sector_buffer + 484) = 0x61417272u;
    *(uint32_t*)(boot_sector_buffer + 488) = 0xFFFFFFFFu;
    *(uint32_t*)(boot_sector_buffer + 492) = 0xFFFFFFFFu;
    *(uint32_t*)(boot_sector_buffer + 508) = 0xAA550000u;
    if (read_write_sectors(g_ahci_port, new_part_lba + new_bpb.fs_info, 1,
                           true, boot_sector_buffer) != 0)
        wm.print_to_focused("Warning: FSInfo sector write failed.\n");
    // Backup FSInfo at bk_boot_sec + 1 (recommended by spec, not required).
    read_write_sectors(g_ahci_port, new_part_lba + new_bpb.bk_boot_sec + 1, 1,
                       true, boot_sector_buffer);

    delete[] boot_sector_buffer;
    g_fs_encryption_enabled = boot_sec_was_enc;

    // ── Clear FAT1 + FAT2 ─────────────────────────────────────────────────
    // fat_start_sector already reflects the correct absolute LBA.
    uint8_t* zero_sector = new uint8_t[SECTOR_SIZE];
    memset(zero_sector, 0, SECTOR_SIZE);
    wm.print_to_focused("Clearing FATs...\n");
    for (uint32_t i = 0; i < bpb.fat_sz32; ++i) {
        read_write_sectors(g_ahci_port, fat_start_sector + i, 1, true, zero_sector);                    // FAT1
        read_write_sectors(g_ahci_port, fat_start_sector + bpb.fat_sz32 + i, 1, true, zero_sector);    // FAT2
    }

    // ── Clear root directory cluster ───────────────────────────────────────
    wm.print_to_focused("Clearing root directory...\n");
    for (uint8_t i = 0; i < bpb.sec_per_clus; ++i) {
        read_write_sectors(g_ahci_port, cluster_to_lba(bpb.root_clus) + i, 1, true, zero_sector);
    }
    delete[] zero_sector;

    // ── Write initial FAT entries ──────────────────────────────────────────
    // Cluster 0: media-type byte (0x0FFFFF_F8 for fixed disk, mirroring BPB_Media).
    // Cluster 1: end-of-chain / dirty-flag word (0x0FFFFFFF = clean).
    // Cluster 2: root directory — single cluster, end of chain.
    wm.print_to_focused("Writing initial FAT entries...\n");
    write_fat_entry(0, 0x0FFFFFF8); // Media descriptor
    write_fat_entry(1, 0x0FFFFFFF); // Clean/EOC
    write_fat_entry(bpb.root_clus, 0x0FFFFFFF); // Root directory EOC

    // ─────────────────────────────────────────────────────────────────────
    // Volume label directory entry in the root cluster.
    //
    // The FAT32 spec (section 6) requires that the first entry in the root
    // directory is a volume-label entry (ATTR_VOLUME_ID = 0x08) whose
    // DIR_Name field matches BPB_VolLab. Without it Windows Explorer shows
    // "Local Disk" instead of the volume name, and chkdsk reports a missing
    // volume label as a warning.
    //
    // The entry is written AFTER the initial FAT entries so that
    // cluster_to_lba(bpb.root_clus) is already valid and fat_start_sector
    // / data_start_sector have been updated for the MBR-partition case.
    // ─────────────────────────────────────────────────────────────────────
    {
        uint8_t* root_sector = new uint8_t[SECTOR_SIZE];
        memset(root_sector, 0, SECTOR_SIZE);
        fat_dir_entry_t* vol_entry = (fat_dir_entry_t*)root_sector;
        memcpy(vol_entry->name, new_bpb.vol_lab, 11); // "MYOS VOL   "
        vol_entry->attr = ATTR_VOLUME_ID;
        // crt_time / crt_date — use a fixed timestamp (2024-01-01 00:00:00).
        // FAT date: bits 15-9 = year-1980, bits 8-5 = month, bits 4-0 = day.
        // FAT time: bits 15-11 = hour, bits 10-5 = minute, bits 4-0 = sec/2.
        vol_entry->crt_date = (uint16_t)((44 << 9) | (1 << 5) | 1); // 2024-01-01
        vol_entry->wrt_date = vol_entry->crt_date;
        vol_entry->lst_acc_date = vol_entry->crt_date;
        vol_entry->fst_clus_hi = 0;
        vol_entry->fst_clus_lo = 0;
        vol_entry->file_size   = 0;
        // Write to the first sector of the root cluster (already zeroed
        // above, but we only wrote zeros — the volume entry goes here now).
        bool vl_ok = (read_write_sectors(
            g_ahci_port, cluster_to_lba(bpb.root_clus), 1, true, root_sector) == 0);
        if (!vl_ok) wm.print_to_focused("Warning: volume label dir entry write failed.\n");
        delete[] root_sector;
        wm.print_to_focused("Volume label directory entry written.\n");
    }

    wm.print_to_focused("Format complete. Re-initializing filesystem...\n");
    if (fat32_init()) {
        wm.print_to_focused("FAT32 FS re-initialized successfully.\n");
    } else {
        wm.print_to_focused("FAT32 FS re-initialization failed.\n");
        // Diagnostic: read the BPB sector back as plaintext and report what
        // is actually on disk. g_partition_lba is the partition's first
        // sector (sector 0 holds the MBR on a raw disk, not the BPB).
        char* vb = new char[SECTOR_SIZE];
        if (vb) {
            bool was = g_fs_encryption_enabled;
            g_fs_encryption_enabled = false;
            int rr = read_write_sectors(g_ahci_port, g_partition_lba, 1, false, vb);
            g_fs_encryption_enabled = was;
            if (rr != 0) {
                wm.print_to_focused("  diag: boot-sector read-back FAILED.\n");
            } else {
                // fil_sys_type sits at BPB offset 82.
                char fst[9];
                for (int k = 0; k < 8; ++k) fst[k] = vb[82 + k];
                fst[8] = '\0';
                uint8_t s0 = (uint8_t)vb[510], s1 = (uint8_t)vb[511];
                char msg[80];
                snprintf(msg, sizeof(msg),
                         "  diag: fs_type='%s' sig=%02X%02X\n",
                         fst, s0, s1);
                wm.print_to_focused(msg);
            }
            delete[] vb;
        }
    }
}
class FileExplorerWindow : public Window {
private:
    char current_path[256];
    fat_dir_entry_t file_list[128];
    int num_files;
    int scroll_offset;
    int selected_index;

public:
    FileExplorerWindow(int x, int y, const char* path) 
        : Window(x, y, 400, 300, "File Explorer"), num_files(0), scroll_offset(0), selected_index(-1) {
        strncpy(current_path, path, 255);
        current_path[255] = '\0';
        refresh_contents();
    }

    void refresh_contents() {
        num_files = fat32_list_directory(current_path, file_list, 128);
    }

    void draw() override {
        if (is_closed) return;
        using namespace ColorPalette;
        
        uint32_t titlebar_color = has_focus ? TITLEBAR_ACTIVE : TITLEBAR_INACTIVE;
        draw_rect_filled(x, y, w, 25, titlebar_color);
        draw_string(title, x + 5, y + 8, TEXT_WHITE);
        draw_string(current_path, x+100, y+8, TEXT_WHITE);

        draw_rect_filled(x + w - 22, y + 4, 18, 18, BUTTON_CLOSE);
        draw_string("X", x + w - 17, y + 8, TEXT_WHITE);
        
        // Main content area
        draw_rect_filled(x, y + 25, w, h - 25, FILE_EXPLORER_BG);
        
        // Draw borders
        for (int i = 0; i < w; i++) put_pixel_back(x + i, y, WINDOW_BORDER);
        for (int i = 0; i < w; i++) put_pixel_back(x + i, y + h - 1, WINDOW_BORDER);
        for (int i = 0; i < h; i++) put_pixel_back(x, y + i, WINDOW_BORDER);
        for (int i = 0; i < h; i++) put_pixel_back(x + w - 1, y + i, WINDOW_BORDER);

        // Draw file list
        int max_visible_items = (h - 35) / 10;
        for (int i = 0; i < max_visible_items; ++i) {
            int file_idx = scroll_offset + i;
            if (file_idx >= num_files) break;
            
            int item_y = y + 30 + i * 10;
            char filename[13];
            fat32_get_fne_from_entry(&file_list[file_idx], filename);

            if (file_idx == selected_index) {
                draw_rect_filled(x + 2, item_y, w - 4, 10, TITLEBAR_ACTIVE);
            }

            if (file_list[file_idx].attr & FAT_ATTR_DIRECTORY) {
                draw_icon_folder(x + 5, item_y - 2);
            } else {
                bool is_shortcut = strstr(filename, ".LNK") != nullptr;
                draw_icon_file(x + 5, item_y - 2, is_shortcut);
            }

            draw_string(filename, x + 40, item_y, TEXT_BLACK);
        }
    }

    void on_key_press(char c) override {
        // Handle keyboard navigation later
    }
void on_mouse_right_click(int mx, int my) {
        int content_y = my - (y + 30);
        if (content_y < 0) return;
        int clicked_idx = scroll_offset + (content_y / 10);

        if (clicked_idx < num_files) {
            selected_index = clicked_idx;
            char filename[13];
            fat32_get_fne_from_entry(&file_list[clicked_idx], filename);

            // Tell the window manager to show the context menu for this file
            wm.show_file_context_menu(mx, my, filename);
        }
    }
    void on_mouse_click(int mx, int my) override {
        int content_y = my - (y + 30);
        if (content_y < 0) return;
        int clicked_idx = scroll_offset + (content_y / 10);
        
        if(clicked_idx < num_files) {
            selected_index = clicked_idx;
            // Basic double-click simulation
            static int last_click_idx = -1;
            static uint32_t last_click_tick = 0;
            if(clicked_idx == last_click_idx && (g_timer_ticks - last_click_tick) < 20) {
                // Double click!
                char filename[13];
                fat32_get_fne_from_entry(&file_list[clicked_idx], filename);

                // --- ADD THIS LOGIC ---
                // Check if it's an executable object file
                if (strstr(filename, ".obj") != nullptr || strstr(filename, ".OBJ") != nullptr) {
                    char command_buffer[128];
                    snprintf(command_buffer, 128, "run %s", filename);
                    launch_terminal_with_command(command_buffer);
                }
                // --- END ADDITION ---

                // Handle opening file/dir (can be expanded for directories later)
            }
            last_click_idx = clicked_idx;
            last_click_tick = g_timer_ticks;
        }
    }

    void update() override {}
};
// ==================== CHKDSK IMPLEMENTATION ====================

struct ChkdskStats {
    uint32_t total_clusters;
    uint32_t used_clusters;
    uint32_t free_clusters;
    uint32_t bad_clusters;
    uint32_t lost_clusters;
    uint32_t directories_checked;
    uint32_t files_checked;
    uint32_t errors_found;
    uint32_t errors_fixed;
};

static uint32_t* cluster_bitmap = nullptr;
static uint32_t cluster_bitmap_size = 0;

void init_cluster_bitmap() {
    uint32_t total_clusters = fat32_max_clusters();
    cluster_bitmap_size = (total_clusters + 31) / 32;
    
    if (cluster_bitmap) delete[] cluster_bitmap;
    cluster_bitmap = new uint32_t[cluster_bitmap_size];
    memset(cluster_bitmap, 0, cluster_bitmap_size * sizeof(uint32_t));
}

void mark_cluster_used(uint32_t cluster) {
    if (cluster < 2) return;
    uint32_t index = cluster / 32;
    uint32_t bit = cluster % 32;
    if (index < cluster_bitmap_size) {
        cluster_bitmap[index] |= (1 << bit);
    }
}

bool is_cluster_marked(uint32_t cluster) {
    if (cluster < 2) return false;
    uint32_t index = cluster / 32;
    uint32_t bit = cluster % 32;
    if (index < cluster_bitmap_size) {
        return (cluster_bitmap[index] & (1 << bit)) != 0;
    }
    return false;
}

bool is_valid_cluster(uint32_t cluster) {
    if (cluster < 2) return false;
    uint32_t max_clusters = fat32_max_clusters();
    return cluster < max_clusters;
}

bool verify_fat_chain(uint32_t start_cluster, uint32_t* chain_length, ChkdskStats& stats) {
    uint32_t current = start_cluster;
    uint32_t count = 0;
    const uint32_t MAX_CHAIN_LENGTH = 1000000;
    
    while (current >= 2 && current < FAT_END_OF_CHAIN && count < MAX_CHAIN_LENGTH) {
        if (!is_valid_cluster(current)) {
            wm.print_to_focused("  ERROR: Invalid cluster in chain!");
            stats.errors_found++;
            return false;
        }
        
        if (is_cluster_marked(current)) {
            wm.print_to_focused("  ERROR: Cross-linked cluster detected!");
            stats.errors_found++;
            return false;
        }
        
        mark_cluster_used(current);
        count++;
        current = read_fat_entry(current);
    }
    
    if (count >= MAX_CHAIN_LENGTH) {
        wm.print_to_focused("  ERROR: Circular FAT chain detected!");
        stats.errors_found++;
        return false;
    }
    
    *chain_length = count;
    return true;
}

bool check_directory_entry(fat_dir_entry_t* entry, ChkdskStats& stats, bool fix) {
    bool has_error = false;
    
    uint32_t start_cluster = (entry->fst_clus_hi << 16) | entry->fst_clus_lo;
    
    if (start_cluster != 0) {
        uint32_t chain_length = 0;
        if (!verify_fat_chain(start_cluster, &chain_length, stats)) {
            has_error = true;
            if (fix) {
                wm.print_to_focused("  FIXING: Truncating bad cluster chain...");
                entry->fst_clus_lo = 0;
                entry->fst_clus_hi = 0;
                entry->file_size = 0;
                stats.errors_fixed++;
            }
        } else {
            uint32_t cluster_size = bpb.sec_per_clus * SECTOR_SIZE;
            uint32_t expected_max_size = chain_length * cluster_size;
            
            if (entry->file_size > expected_max_size) {
                wm.print_to_focused("  ERROR: File size exceeds allocated clusters!");
                stats.errors_found++;
                has_error = true;
                
                if (fix) {
                    entry->file_size = expected_max_size;
                    wm.print_to_focused("  FIXED: Corrected file size");
                    stats.errors_fixed++;
                }
            }
        }
    } else if (entry->file_size != 0) {
        wm.print_to_focused("  ERROR: File has size but no cluster allocation!");
        stats.errors_found++;
        has_error = true;
        
        if (fix) {
            entry->file_size = 0;
            wm.print_to_focused("  FIXED: Reset file size to 0");
            stats.errors_fixed++;
        }
    }
    
    return !has_error;
}

bool scan_directory(uint32_t start_cluster, ChkdskStats& stats, bool fix, int depth = 0) {
    if (depth > 20) {
        wm.print_to_focused("ERROR: Directory nesting too deep!");
        return false;
    }
    
    stats.directories_checked++;

    uint32_t cluster_bytes = bpb.sec_per_clus * SECTOR_SIZE;
    uint8_t* buffer = new uint8_t[cluster_bytes];

    // Create a working copy for modifications
    uint8_t* working_buffer = nullptr;
    if (fix) {
        working_buffer = new uint8_t[cluster_bytes];
    }

    bool ok = true;

    // A directory is an ordinary cluster chain, exactly like a file. The
    // previous version of this function only ever looked at the FIRST
    // cluster of the chain, so any entries that spilled into a second or
    // later cluster (trivial to hit: a 512-byte cluster holds only 16
    // entries) were never scanned, never marked "in use" in the cluster
    // bitmap, and their files never appeared as checked by chkdsk. This
    // is the same root cause behind files copied in from another OS
    // going missing from `ls` — we now walk the whole chain, stopping
    // only at a genuine end-of-directory marker (name[0] == 0x00).
    uint32_t cluster = start_cluster;
    bool end_of_dir = false;
    while (!end_of_dir && cluster >= 2 && cluster < FAT_END_OF_CHAIN) {
        mark_cluster_used(cluster);

        if (read_write_sectors(g_ahci_port, cluster_to_lba(cluster), bpb.sec_per_clus, false, buffer) != 0) {
            wm.print_to_focused("ERROR: Cannot read directory cluster");
            ok = false;
            break;
        }

        if (fix) memcpy(working_buffer, buffer, cluster_bytes);
        bool modified = false;

        for (uint32_t i = 0; i < cluster_bytes; i += sizeof(fat_dir_entry_t)) {
            // Use working buffer if fixing, otherwise use read-only buffer
            fat_dir_entry_t* entry = (fat_dir_entry_t*)((fix ? working_buffer : buffer) + i);

            if (entry->name[0] == 0x00) { end_of_dir = true; break; }
            if ((uint8_t)entry->name[0] == DELETED_ENTRY) continue;
            if (entry->name[0] == '.') continue;

            if (entry->attr == ATTR_LONG_NAME) continue;
            if (entry->attr & ATTR_VOLUME_ID) continue;

            stats.files_checked++;

            char fname[13];
            from_83_format(entry->name, fname);

            char msg[100];
            snprintf(msg, 100, "Checking: %s", fname);
            wm.print_to_focused(msg);

            // Only mark as modified if we're in fix mode and something changed
            if (!check_directory_entry(entry, stats, fix)) {
                if (fix) {
                    modified = true;
                }
            }

            if (entry->attr & 0x10) {
                uint32_t subcluster = (entry->fst_clus_hi << 16) | entry->fst_clus_lo;
                if (subcluster >= 2 && subcluster < FAT_END_OF_CHAIN) {
                    if (!is_cluster_marked(subcluster)) {
                        scan_directory(subcluster, stats, fix, depth + 1);
                    }
                }
            }
        }

        // ONLY write back if in fix mode AND something was modified
        if (fix && modified) {
            read_write_sectors(g_ahci_port, cluster_to_lba(cluster), bpb.sec_per_clus, true, working_buffer);
        }

        if (!end_of_dir) cluster = read_fat_entry(cluster);
    }

    delete[] buffer;
    if (working_buffer) {
        delete[] working_buffer;
    }

    return ok;
}


void find_lost_clusters(ChkdskStats& stats, bool fix) {
    wm.print_to_focused("\nScanning for lost clusters...");
    
    uint32_t max_clusters = fat32_max_clusters();
    
    for (uint32_t cluster = 2; cluster < max_clusters; cluster++) {
        uint32_t fat_entry = read_fat_entry(cluster);
        
        if (fat_entry != FAT_FREE_CLUSTER && !is_cluster_marked(cluster)) {
            stats.lost_clusters++;
            
            char msg[80];
            snprintf(msg, 80, "  Lost cluster chain starting at %d", cluster);
            wm.print_to_focused(msg);
            
            if (fix) {
                uint32_t current = cluster;
                while (current >= 2 && current < FAT_END_OF_CHAIN) {
                    uint32_t next = read_fat_entry(current);
                    write_fat_entry(current, FAT_FREE_CLUSTER);
                    current = next;
                    stats.errors_fixed++;
                }
                wm.print_to_focused("  FIXED: Freed lost cluster chain");
            }
        }
    }
}

bool check_fat_consistency(ChkdskStats& stats, bool fix) {
    wm.print_to_focused("Checking FAT table consistency...");
    
    if (bpb.num_fats < 2) {
        wm.print_to_focused("WARNING: Only one FAT copy present!");
        return true;
    }
    
    uint32_t fat_size = bpb.fat_sz32 * SECTOR_SIZE;
    uint8_t* fat1 = new uint8_t[fat_size];
    uint8_t* fat2 = new uint8_t[fat_size];
    
    read_write_sectors(g_ahci_port, fat_start_sector, bpb.fat_sz32, false, fat1);
    read_write_sectors(g_ahci_port, fat_start_sector + bpb.fat_sz32, bpb.fat_sz32, false, fat2);
    
    bool mismatch = false;
    for (uint32_t i = 0; i < fat_size; i++) {
        if (fat1[i] != fat2[i]) {
            mismatch = true;
            break;
        }
    }
    
    if (mismatch) {
        wm.print_to_focused("ERROR: FAT1 and FAT2 do not match!");
        stats.errors_found++;
        
        if (fix) {
            wm.print_to_focused("FIXING: Copying FAT1 to FAT2...");
            read_write_sectors(g_ahci_port, fat_start_sector + bpb.fat_sz32, bpb.fat_sz32, true, fat1);
            stats.errors_fixed++;
            wm.print_to_focused("FIXED: FAT tables synchronized");
        }
    } else {
        wm.print_to_focused("OK: FAT tables are consistent");
    }
    
    delete[] fat1;
    delete[] fat2;
    return !mismatch;
}
void chkdsk(bool fix = false, bool verbose = false) {
    // Safety check
    if (!ahci_base || !current_directory_cluster) {
        wm.print_to_focused("ERROR: Filesystem not initialized!");
        return;
    }
    
    wm.print_to_focused("=====================================");
    wm.print_to_focused("    DISK CHECK UTILITY (CHKDSK)     ");
    wm.print_to_focused("=====================================");
    
    if (fix) {
        wm.print_to_focused("\nMode: FIX ERRORS (writing enabled)");
    } else {
        wm.print_to_focused("\nMode: READ-ONLY (no changes)");
    }
    
    ChkdskStats stats;
    memset(&stats, 0, sizeof(stats));
    
    // SAFETY: Check for valid values
    if (bpb.sec_per_clus == 0) {
        wm.print_to_focused("ERROR: Invalid cluster size!");
        return;
    }
    
    if (bpb.tot_sec32 <= bpb.rsvd_sec_cnt + (uint32_t)bpb.num_fats * bpb.fat_sz32) {
        wm.print_to_focused("ERROR: Invalid disk geometry!");
        return;
    }
    
    stats.total_clusters = fat32_data_sectors() / bpb.sec_per_clus;
    
    // SAFETY: Prevent division by zero
    if (stats.total_clusters == 0) {
        wm.print_to_focused("ERROR: No data clusters available!");
        return;
    }
    
    char msg[100];
    snprintf(msg, 100, "\nVolume size: %d sectors (%d MB)", 
             bpb.tot_sec32, (bpb.tot_sec32 * SECTOR_SIZE) / (1024 * 1024));
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Cluster size: %d KB", (bpb.sec_per_clus * SECTOR_SIZE) / 1024);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Total clusters: %d", stats.total_clusters);
    wm.print_to_focused(msg);
    
    wm.print_to_focused("\n=== Phase 1: Checking boot sector ===");
    
    if (strncmp(bpb.fil_sys_type, "FAT32   ", 8) != 0) {
        wm.print_to_focused("ERROR: Invalid filesystem type!");
        return;
    }
    wm.print_to_focused("OK: Boot sector is valid");
    
    // Comment out FAT consistency check for now (might be causing issue)
    // check_fat_consistency(stats, fix);
    
    wm.print_to_focused("\n=== Phase 2: Scanning directories ===");
    
    // SAFETY: Pre-flight the cluster bitmap allocation.
    //
    // init_cluster_bitmap() needs 1 bit per cluster on the volume. The
    // check below (`if (!cluster_bitmap)`) looks like it handles an
    // allocation failure, but it never actually can: `operator new[]`
    // in this kernel does not return null on failure — it calls
    // oom_halt(), which paints an error to the screen and then HALTS
    // THE WHOLE KERNEL (cli; hlt loop). So on any volume big enough
    // (or with corrupt/garbage BPB geometry — easy to end up with after
    // the disk has been repartitioned/reformatted by another OS) that
    // the bitmap doesn't fit in whatever's left of the heap, chkdsk
    // didn't fail cleanly — it froze the entire machine. That's the
    // "chkdsk causes an OOM crash" bug.
    //
    // Fix: compute the bitmap size up front and compare it against
    // g_allocator.total_free() (with a safety margin for everything
    // else still running — windows, the terminal's own buffers, other
    // in-flight FAT32 operations) *before* calling operator new. If it
    // won't fit, report a normal chkdsk error and return instead of
    // ever reaching the allocation that would halt the kernel.
    {
        uint64_t clusters_for_bitmap = (uint64_t)fat32_max_clusters();
        uint64_t bitmap_bytes = ((clusters_for_bitmap + 31) / 32) * sizeof(uint32_t);
        const uint64_t SAFETY_MARGIN = 4 * 1024 * 1024; // leave 4MB free for everything else
        uint64_t free_now = (uint64_t)g_allocator.total_free();

        if (bitmap_bytes + SAFETY_MARGIN > free_now) {
            // NOTE: this kernel's snprintf only implements %d/%s/%c (no
            // %u/%llu/%x), so keep everything as plain `int` KB counts.
            // Both values are bounded well within INT32_MAX for any
            // heap size this kernel actually uses (tens of MB), even
            // for a maximally-corrupt uint32_t cluster count.
            int needed_kb = (int)(bitmap_bytes / 1024);
            int free_kb   = (int)(free_now / 1024);
            snprintf(msg, 100, "ERROR: Volume too large for chkdsk (needs %d KB,", needed_kb);
            wm.print_to_focused(msg);
            snprintf(msg, 100, "       only %d KB free). Aborting safely.", free_kb);
            wm.print_to_focused(msg);
            return;
        }
    }

    // SAFETY: Initialize bitmap
    init_cluster_bitmap();
    if (!cluster_bitmap) {
        wm.print_to_focused("ERROR: Failed to allocate cluster bitmap!");
        return;
    }
    
    mark_cluster_used(0);
    mark_cluster_used(1);
    
    // SAFETY: Check root cluster validity
    if (bpb.root_clus < 2 || bpb.root_clus >= FAT_END_OF_CHAIN) {
        wm.print_to_focused("ERROR: Invalid root cluster!");
        if (cluster_bitmap) {
            delete[] cluster_bitmap;
            cluster_bitmap = nullptr;
        }
        return;
    }
    
    mark_cluster_used(bpb.root_clus);
    
    wm.print_to_focused("Scanning root directory...");
    
    // SAFETY: Limit recursion depth to prevent stack overflow
    scan_directory(bpb.root_clus, stats, fix, 0);
    
    wm.print_to_focused("\n=== Phase 3: Statistics ===");
    
    // Simple stats without lost cluster scan (can add back later)
    for (uint32_t i = 2; i < stats.total_clusters + 2; i++) {
        uint32_t entry = read_fat_entry(i);
        if (entry == FAT_FREE_CLUSTER) {
            stats.free_clusters++;
        } else if (entry >= 0x0FFFFFF7) {
            stats.bad_clusters++;
        } else {
            stats.used_clusters++;
        }
    }
    
    wm.print_to_focused("\n=====================================");
    wm.print_to_focused("         CHKDSK RESULTS              ");
    wm.print_to_focused("=====================================");
    
    snprintf(msg, 100, "Directories checked:  %d", stats.directories_checked);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Files checked:        %d", stats.files_checked);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "\nTotal clusters:       %d", stats.total_clusters);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Used clusters:        %d (%d%%)", 
             stats.used_clusters, (stats.used_clusters * 100) / stats.total_clusters);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Free clusters:        %d (%d%%)", 
             stats.free_clusters, (stats.free_clusters * 100) / stats.total_clusters);
    wm.print_to_focused(msg);
    
    snprintf(msg, 100, "Bad clusters:         %d", stats.bad_clusters);
    wm.print_to_focused(msg);
    
    wm.print_to_focused("");
    snprintf(msg, 100, "Errors found:         %d", stats.errors_found);
    wm.print_to_focused(msg);
    
    if (fix && stats.errors_fixed > 0) {
        snprintf(msg, 100, "Errors fixed:         %d", stats.errors_fixed);
        wm.print_to_focused(msg);
    }
    
    if (stats.errors_found == 0) {
        wm.print_to_focused("\nNo errors found. Disk is healthy!");
    }
    
    // Cleanup
    if (cluster_bitmap) {
        delete[] cluster_bitmap;
        cluster_bitmap = nullptr;
    }
    
    wm.print_to_focused("=====================================");
}


void chkdsk_full_scan(bool fix = false) {
    wm.print_to_focused("\n=== Phase 5: Scanning for bad sectors ===");
    wm.print_to_focused("This may take several minutes...");
    
    uint8_t* test_buffer = new uint8_t[SECTOR_SIZE];
    uint32_t bad_sectors = 0;
    uint32_t total_sectors = bpb.tot_sec32;
    
    for (uint32_t sector = 0; sector < total_sectors; sector += 1) {
        if (read_write_sectors(g_ahci_port, sector, 1, false, test_buffer) != 0) {
            bad_sectors++;
            
            char msg[80];
            snprintf(msg, 80, "  Bad sector detected at LBA %d", sector);
            wm.print_to_focused(msg);
            
            if (sector >= data_start_sector) {
                uint32_t cluster = ((sector - data_start_sector) / bpb.sec_per_clus) + 2;
                if (fix && is_valid_cluster(cluster)) {
                    write_fat_entry(cluster, 0x0FFFFFF7);
                    wm.print_to_focused("  FIXED: Marked cluster as bad in FAT");
                }
            }
        }
        
        if ((sector / 1000) % 10 == 0 && sector > 0) {
            char progress[60];
            snprintf(progress, 60, "Progress: %d%% (%d/%d sectors)", 
                     (sector * 100) / total_sectors, sector, total_sectors);
            wm.print_to_focused(progress);
        }
    }
    
    delete[] test_buffer;
    
    char summary[80];
    snprintf(summary, 80, "\nBad sector scan complete: %d bad sectors found", bad_sectors);
    wm.print_to_focused(summary);
}


#include <cstdarg>    // For va_list in printf

// =============================================================================
// SECTION 6: SELF-HOSTED C COMPILER
// =============================================================================

// Forward declarations consumed by the command shell
extern "C" void cmd_compile(uint64_t ahci_base, int port, const char* filename);
extern "C" void cmd_run(uint64_t ahci_base, int port, const char* filename);
extern "C" void cmd_exec(const char* code_text);
struct HardwareDevice {
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t base_address;
    uint64_t size;
    uint32_t device_type;  // 0=Unknown, 1=Storage, 2=Network, 3=Graphics, 4=Audio, 5=USB
    char description[64];
};
// --- Global Hardware Registry Definition ---
const int MAX_HARDWARE_DEVICES = 32; // Define the constant
HardwareDevice hardware_registry[MAX_HARDWARE_DEVICES];
int hardware_count = 0;

// Define shell parts variables (as declared extern in the header)
// These will be populated by the terminal handler in kernel.cpp
char* parts[32];
int   part_count = 0;


// ---- tiny helpers ----
static inline int tcc_is_digit(char c){ return c>='0' && c<='9'; }
static inline int tcc_is_alpha(char c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'; }
static inline int tcc_is_alnum(char c){ return tcc_is_alpha(c)||tcc_is_digit(c); }
static inline int tcc_strlen(const char* s){ int n=0; while(s && s[n]) ++n; return n; }

// ============================================================
// Console and Terminal I/O Functions
// ============================================================
void console_putc(char c) {
    wm.put_char_to_focused(c);
}
// VGA Text Mode Buffer (typically at 0xB8000)
static volatile char* const VGA_BUFFER = (volatile char* const)0xB8000;
static int vga_row = 0;
static int vga_col = 0;
static const int VGA_WIDTH = 80;
static const int VGA_HEIGHT = 23;
void vga_print_char(char c) {
    if (c == '\n') {
        vga_col = 0;
        vga_row++;
        if (vga_row >= VGA_HEIGHT) {
            vga_row = VGA_HEIGHT - 1;
            // Scroll VGA buffer up
            for (int row = 0; row < VGA_HEIGHT - 1; row++) {
                for (int col = 0; col < VGA_WIDTH; col++) {
                    int src_idx = ((row + 1) * VGA_WIDTH + col) * 2;
                    int dst_idx = (row * VGA_WIDTH + col) * 2;
                    VGA_BUFFER[dst_idx] = VGA_BUFFER[src_idx];
                    VGA_BUFFER[dst_idx + 1] = VGA_BUFFER[src_idx + 1];
                }
            }
            // Clear last line
            for (int col = 0; col < VGA_WIDTH; col++) {
                int idx = ((VGA_HEIGHT - 1) * VGA_WIDTH + col) * 2;
                VGA_BUFFER[idx] = ' ';
                VGA_BUFFER[idx + 1] = 0x07;
            }
        }
    } else if (c >= 32 && c < 127) {
        int index = (vga_row * VGA_WIDTH + vga_col) * 2;
        VGA_BUFFER[index] = c;
        VGA_BUFFER[index + 1] = 0x07;
        vga_col++;
        if (vga_col >= VGA_WIDTH) {
            vga_col = 0;
            vga_row++;
            if (vga_row >= VGA_HEIGHT) {
                vga_row = VGA_HEIGHT - 1;
                // Scroll VGA buffer up
                for (int row = 0; row < VGA_HEIGHT - 1; row++) {
                    for (int col = 0; col < VGA_WIDTH; col++) {
                        int src_idx = ((row + 1) * VGA_WIDTH + col) * 2;
                        int dst_idx = (row * VGA_WIDTH + col) * 2;
                        VGA_BUFFER[dst_idx] = VGA_BUFFER[src_idx];
                        VGA_BUFFER[dst_idx + 1] = VGA_BUFFER[src_idx + 1];
                    }
                }
                // Clear last line
                for (int col = 0; col < VGA_WIDTH; col++) {
                    int idx = ((VGA_HEIGHT - 1) * VGA_WIDTH + col) * 2;
                    VGA_BUFFER[idx] = ' ';
                    VGA_BUFFER[idx + 1] = 0x07;
                }
            }
        }
    }
}

void vga_print(const char* str) {
    if (!str) return;
    while (*str) {
        vga_print_char(*str);
        str++;
    }
}

// Route to window if available, otherwise VGA
void console_print_char(char c) {
    int num_wins = wm.get_num_windows();
    int focused = wm.get_focused_idx();
    if (num_wins > 0 && focused >= 0 && focused < num_wins) {
        Window* win = wm.get_window(focused);
        if (win) {
            char buf[2] = {c, 0};
            win->console_print(buf);
        }
    } else {
        vga_print_char(c);
    }
}

void console_print(const char* str) {
    if (!str) return;
    int num_wins = wm.get_num_windows();
    int focused = wm.get_focused_idx();
    if (num_wins > 0 && focused >= 0 && focused < num_wins) {
        Window* win = wm.get_window(focused);
        if (win) {
            win->console_print(str);
        }
    } else {
        vga_print(str);
    }
}

// CORRECTED: Non-blocking get_char with fallback
static char pending_char = 0;

char get_char() {
    // Check if we have a pending character from previous call
    if (pending_char != 0) {
        char c = pending_char;
        pending_char = 0;
        return c;
    }

    // Non-blocking read from keyboard
    while (1) {
        uint8_t status = inb(0x64);
        if (status & 0x01) { // Data available
            uint8_t scancode = inb(0x60);

            // Simple scancode to ASCII conversion (US keyboard layout)
            static const char scancode_map[] = {
                0,   27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b', '\t',
                'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\n', 0, 'a', 's',
                'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0, '\\', 'z', 'x', 'c', 'v',
                'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '
            };

            if (scancode < sizeof(scancode_map)) {
                char c = scancode_map[scancode];
                if (c != 0) {
                    vga_print_char(c);
                    return c;
                }
            }
        } else {
            // No data available - return a null character
            // The caller should handle this and retry if needed
            return 0;
        }
    }
}


// ============================================================
// Integer Conversion Functions
// ============================================================

void int_to_string(int value, char* buffer) {
    if (!buffer) return;
    
    if (value == 0) {
        buffer[0] = '0';
        buffer[1] = 0;
        return;
    }
    
    int negative = value < 0;
    if (negative) value = -value;
    
    int i = 0;
    char temp[16];
    
    while (value > 0) {
        temp[i++] = '0' + (value % 10);
        value /= 10;
    }
    
    int j = 0;
    if (negative) buffer[j++] = '-';
    
    while (i > 0) {
        buffer[j++] = temp[--i];
    }
    
    buffer[j] = 0;
}


// ============================================================
// File I/O Functions (FAT32 Support)
// ============================================================

// Simplified file buffer for storage
static char file_buffer[4096]; // 4KB — stub buffer (real I/O goes through heap)


// ============================================================
// Memory Management (new/delete operators)
// ============================================================

// (heap managed by g_allocator via kernel_heap[] above)


void simple_strcpy(char* dest, const char* src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\0';
}

int simple_strcmp(const char* s1, const char* s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

void* simple_memcpy(void* dest, const void* src, int n) {
    char* d = (char*)dest;
    const char* s = (const char*)src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}
// Basic printf implementation
void printf(const char* format, ...) {
    va_list args;
    va_start(args, format);

    char buffer[256]; // A buffer to hold consecutive characters
    int buffer_index = 0;

    while (*format != '\0') {
        if (*format == '%') {
            // If there's anything in the buffer, print it first
            if (buffer_index > 0) {
                buffer[buffer_index] = '\0';
                console_print(buffer);
                buffer_index = 0; // Reset buffer
            }

            format++; // Move past the '%'
            if (*format == 'd') {
                int i = va_arg(args, int);
                char num_buf[12];
                int_to_string(i, num_buf);
                console_print(num_buf);
            } else if (*format == 's') {
                char* s = va_arg(args, char*);
                console_print(s);
            } else if (*format == 'c') {
                char c = (char)va_arg(args, int);
                char str[2] = {c, 0};
                console_print(str);
            } else { // Handles %% and unknown specifiers
                console_print_char('%');
                console_print_char(*format);
            }
        } else {
            // Add the character to our buffer
            if (buffer_index < 255) {
                buffer[buffer_index++] = *format;
            }
        }
        format++;
    }

    // Print any remaining characters in the buffer at the end
    if (buffer_index > 0) {
        buffer[buffer_index] = '\0';
        console_print(buffer);
    }

    va_end(args);
}


// Helper functions for hex conversion and PCI access
static void uint32_to_hex_string(uint32_t value, char* buffer) {
    const char hex_chars[] = "0123456789ABCDEF";
    for(int i = 7; i >= 0; i--) {
        buffer[7-i] = hex_chars[(value >> (i*4)) & 0xF];
    }
    buffer[8] = 0;
}

static void uint64_to_hex_string(uint64_t value, char* buffer) {
    const char hex_chars[] = "0123456789ABCDEF";
    for(int i = 15; i >= 0; i--) {
        buffer[15-i] = hex_chars[(value >> (i*4)) & 0xF];
    }
    buffer[16] = 0;
}

// Simple PCI configuration space access
static uint32_t pci_config_read_dword(uint16_t bus, uint8_t device, uint8_t function, uint8_t offset) {
    uint32_t address = 0x80000000 | ((uint32_t)bus << 16) | ((uint32_t)device << 11) |
                       ((uint32_t)function << 8) | (offset & 0xFC);

    // Write address to CONFIG_ADDRESS (0xCF8)
    asm volatile("outl %0, %w1" : : "a"(address), "Nd"(0xCF8) : "memory");

    // Read data from CONFIG_DATA (0xCFC)
    uint32_t result;
    asm volatile("inl %w1, %0" : "=a"(result) : "Nd"(0xCFC) : "memory");

    return result;
}



// Global hardware_registry and hardware_count are defined at the top of the file.

// More comprehensive PCI class codes
static const char* get_pci_class_name(uint8_t base_class, uint8_t sub_class) {
    switch (base_class) {
        case 0x00: return "Unclassified";
        case 0x01:
            switch (sub_class) {
                case 0x00: return "SCSI Controller";
                case 0x01: return "IDE Controller";
                case 0x02: return "Floppy Controller";
                case 0x03: return "IPI Controller";
                case 0x04: return "RAID Controller";
                case 0x05: return "ATA Controller";
                case 0x06: return "SATA Controller";
                case 0x07: return "SAS Controller";
                case 0x08: return "NVMe Controller";
                default: return "Storage Controller";
            }
        case 0x02: return "Network Controller";
        case 0x03:
            switch (sub_class) {
                case 0x00: return "VGA Controller";
                case 0x01: return "XGA Controller";
                case 0x02: return "3D Controller";
                default: return "Display Controller";
            }
        case 0x04: return "Multimedia Controller";
        case 0x05: return "Memory Controller";
        case 0x06: return "Bridge Device";
        case 0x07: return "Communication Controller";
        case 0x08: return "System Peripheral";
        case 0x09: return "Input Device";
        case 0x0A: return "Docking Station";
        case 0x0B: return "Processor";
        case 0x0C:
            switch (sub_class) {
                case 0x00: return "FireWire Controller";
                case 0x01: return "ACCESS Bus";
                case 0x02: return "SSA";
                case 0x03: return "USB Controller";
                case 0x04: return "Fibre Channel";
                case 0x05: return "SMBus";
                default: return "Serial Bus Controller";
            }
        case 0x0D: return "Wireless Controller";
        case 0x0E: return "Intelligent Controller";
        case 0x0F: return "Satellite Controller";
        case 0x10: return "Encryption Controller";
        case 0x11: return "Signal Processing Controller";
        default: return "Unknown Device";
    }
}

// Improved PCI device discovery
static void discover_pci_devices() {
    for (uint16_t bus = 0; bus < 256; bus++) {
        for (uint8_t device = 0; device < 32; device++) {
            for (uint8_t function = 0; function < 8; function++) {
                uint32_t vendor_device = pci_config_read_dword(bus, device, function, 0);
                if ((vendor_device & 0xFFFF) == 0xFFFF) continue;

                if (hardware_count >= MAX_HARDWARE_DEVICES) return;

                HardwareDevice& dev = hardware_registry[hardware_count];
                dev.vendor_id = vendor_device & 0xFFFF;
                dev.device_id = (vendor_device >> 16) & 0xFFFF;

                // Read class code
                uint32_t class_code = pci_config_read_dword(bus, device, function, 0x08);
                uint8_t base_class = (class_code >> 24) & 0xFF;
                uint8_t sub_class = (class_code >> 16) & 0xFF;

                // Map to device type
                switch (base_class) {
                    case 0x01: dev.device_type = 1; break; // Storage
                    case 0x02: dev.device_type = 2; break; // Network
                    case 0x03: dev.device_type = 3; break; // Graphics
                    case 0x04: dev.device_type = 4; break; // Audio
                    case 0x0C:
                        dev.device_type = (sub_class == 0x03) ? 5 : 0; // USB or other
                        break;
                    default: dev.device_type = 0; break;
                }

                // Get description
                const char* desc = get_pci_class_name(base_class, sub_class);
                strncpy(dev.description, desc, 63);
                dev.description[63] = '\0';

                // Read BAR0 for base address (handle both 32-bit and 64-bit BARs)
                uint32_t bar0 = pci_config_read_dword(bus, device, function, 0x10);
                if (bar0 & 0x1) {
                    // I/O port
                    dev.base_address = bar0 & 0xFFFFFFFC;
                    dev.size = 0x100;
                } else {
                    // Memory mapped
                    dev.base_address = bar0 & 0xFFFFFFF0;
                    
                    // Check if 64-bit BAR
                    if ((bar0 & 0x6) == 0x4) {
                        uint32_t bar1 = pci_config_read_dword(bus, device, function, 0x14);
                        dev.base_address |= ((uint64_t)bar1 << 32);
                    }
                    
                    // Try to determine size by writing all 1s and reading back
                    pci_config_read_dword(bus, device, function, 0x04); // Save command reg
                    uint32_t orig_bar = bar0;
                    
                    outl(0xCF8, 0x80000000 | ((uint32_t)bus << 16) | 
                         ((uint32_t)device << 11) | ((uint32_t)function << 8) | 0x10);
                    outl(0xCFC, 0xFFFFFFFF);
                    uint32_t size_bar = inl(0xCFC);
                    
                    // Restore original BAR
                    outl(0xCF8, 0x80000000 | ((uint32_t)bus << 16) | 
                         ((uint32_t)device << 11) | ((uint32_t)function << 8) | 0x10);
                    outl(0xCFC, orig_bar);
                    
                    if (size_bar != 0 && size_bar != 0xFFFFFFFF) {
                        size_bar &= 0xFFFFFFF0;
                        dev.size = ~size_bar + 1;
                    } else {
                        dev.size = 0x1000; // Default to 4KB
                    }
                }

                hardware_count++;

                if (function == 0) {
                    uint8_t header_type = (class_code >> 16) & 0xFF;
                    if (!(header_type & 0x80)) {
                        break; // Single function device
                    }
                }
            }
        }
    }
}


static void discover_memory_regions() {
    // Add known memory regions
    if (hardware_count < MAX_HARDWARE_DEVICES) {
        HardwareDevice& dev = hardware_registry[hardware_count];
        dev.vendor_id = 0x0000;
        dev.device_id = 0x0001;
        dev.base_address = 0xB8000; // VGA text mode buffer
        dev.size = 0x8000;
        dev.device_type = 3;
        simple_strcpy(dev.description, "VGA Text Buffer");
        hardware_count++;
    }

    if (hardware_count < MAX_HARDWARE_DEVICES) {
        HardwareDevice& dev = hardware_registry[hardware_count];
        dev.vendor_id = 0x0000;
        dev.device_id = 0x0002;
        dev.base_address = 0xA0000; // VGA graphics buffer
        dev.size = 0x20000;
        dev.device_type = 3;
        simple_strcpy(dev.description, "VGA Graphics Buffer");
        hardware_count++;
    }
}

static int scan_hardware() {
    hardware_count = 0;
    discover_pci_devices();
    discover_memory_regions();
    return hardware_count;
}

// Safety check for MMIO access
static bool is_safe_mmio_address(uint64_t addr, uint64_t size) {
    // Check if address falls within any known device range
    for (int i = 0; i < hardware_count; i++) {
        const HardwareDevice& dev = hardware_registry[i];
        if (addr >= dev.base_address &&
            addr + size <= dev.base_address + dev.size) {
            return true;
        }
    }

    // Allow access to standard VGA and system areas even if not enumerated
    if (addr >= 0xA0000 && addr < 0x100000) return true; // VGA/BIOS area
    if (addr >= 0xB8000 && addr < 0xC0000) return true; // VGA text buffer
    if (addr >= 0x3C0 && addr < 0x3E0) return true;     // VGA registers
    if (addr >= 0x60 && addr < 0x70) return true;       // Keyboard controller

    return false;
}

// ============================================================
// Enhanced Bytecode ISA with Hardware Discovery and MMIO
// ============================================================
enum TOp : unsigned char {
    // stack/data
    T_NOP=0, T_PUSH_IMM, T_PUSH_STR, T_LOAD_LOCAL, T_STORE_LOCAL, T_POP,

    // arithmetic / unary
    T_ADD, T_SUB, T_MUL, T_DIV, T_NEG,

    // comparisons
    T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,

    // control flow
    T_JMP, T_JZ, T_JNZ, T_RET,

    // I/O and args
    T_PRINT_INT, T_PRINT_CHAR, T_PRINT_STR, T_PRINT_ENDL, T_PRINT_INT_ARRAY, T_PRINT_STRING_ARRAY,
    T_READ_INT, T_READ_CHAR, T_READ_STR,
    T_PUSH_ARGC, T_PUSH_ARGV_PTR,

    // File I/O operations
    T_READ_FILE, T_WRITE_FILE, T_APPEND_FILE,

    // Array operations
    T_ALLOC_ARRAY, T_LOAD_ARRAY, T_STORE_ARRAY, T_ARRAY_SIZE, T_ARRAY_RESIZE,

    // String operations
    T_STR_CONCAT, T_STR_LENGTH, T_STR_SUBSTR, T_INT_TO_STR, T_STR_COMPARE,
    T_STR_FIND_CHAR, T_STR_FIND_STR, T_STR_FIND_LAST_CHAR, T_STR_CONTAINS,
    T_STR_STARTS_WITH, T_STR_ENDS_WITH, T_STR_COUNT_CHAR, T_STR_REPLACE_CHAR,

    // NEW: Hardware Discovery and Memory-Mapped I/O
    T_SCAN_HARDWARE,      // () -> device_count
    T_GET_DEVICE_INFO,    // (device_index) -> device_array_handle
    T_MMIO_READ8,         // (address) -> uint8_value
    T_MMIO_READ16,        // (address) -> uint16_value
    T_MMIO_READ32,        // (address) -> uint32_value
    T_MMIO_READ64,        // (address) -> uint64_value (split into two 32-bit values)
    T_MMIO_WRITE8,        // (address, value) -> success
    T_MMIO_WRITE16,       // (address, value) -> success
    T_MMIO_WRITE32,       // (address, value) -> success
    T_MMIO_WRITE64,       // (address, low32, high32) -> success
    T_GET_HARDWARE_ARRAY, // () -> hardware_device_array_handle
    T_DISPLAY_MEMORY_MAP // () -> displays formatted memory map
};

// ============================================================
// Enhanced Program buffers with hardware support
// ============================================================
struct TProgram {
    static const int CODE_MAX = 8192;
    unsigned char code[CODE_MAX];
    int pc = 0;

    static const int LIT_MAX = 4096;
    char lit[LIT_MAX];
    int lit_top = 0;

    static const int LOC_MAX = 32;
    char  loc_name[LOC_MAX][32];
    unsigned char loc_type[LOC_MAX]; // 0=int,1=char,2=string,3=int_array,4=string_array,5=device_array
    int   loc_array_size[LOC_MAX];
    int   loc_count = 0;

    int add_local(const char* name, unsigned char t, int array_size = 0){
        for(int i=0;i<loc_count;i++){ if(simple_strcmp(loc_name[i], name)==0) return i; }
        if(loc_count>=LOC_MAX) return -1;
        simple_strcpy(loc_name[loc_count], name);
        loc_type[loc_count]=t;
        loc_array_size[loc_count] = array_size;
        return loc_count++;
    }
    int get_local(const char* name){
        for(int i=0;i<loc_count;i++){ if(simple_strcmp(loc_name[i], name)==0) return i; }
        return -1;
    }
    int get_local_type(int idx){ return (idx>=0 && idx<loc_count)? loc_type[idx] : 0; }
    int get_array_size(int idx){ return (idx>=0 && idx<loc_count)? loc_array_size[idx] : 0; }

    void emit1(unsigned char op){ if(pc<CODE_MAX) code[pc++]=op; }
    void emit4(int v){ if(pc+4<=CODE_MAX){ code[pc++]=v&0xff; code[pc++]=(v>>8)&0xff; code[pc++]=(v>>16)&0xff; code[pc++]=(v>>24)&0xff; } }
    int  mark(){ return pc; }
    void patch4(int at, int v){ if(at+4<=CODE_MAX){ code[at+0]=v&0xff; code[at+1]=(v>>8)&0xff; code[at+2]=(v>>16)&0xff; code[at+3]=(v>>24)&0xff; } }

    const char* add_lit(const char* s){
        int n = tcc_strlen(s)+1;
        if(lit_top+n > LIT_MAX) return "";
        char* p = &lit[lit_top];
        simple_memcpy(p, s, n);
        lit_top += n;
        return p;
    }
};

// ============================================================
// Enhanced Tokenizer with hardware and MMIO keywords
// ============================================================
enum TTokType { TT_EOF, TT_ID, TT_NUM, TT_STR, TT_CH, TT_KW, TT_OP, TT_PUNC };
struct TTok { TTokType t; char v[256]; int ival; };

struct TLex {
    const char* src; int pos; int line;
    void init(const char* s){ src=s; pos=0; line=1; }

    void skipws(){
        for(;;){
            char c=src[pos];
            if(c==' '||c=='\t'||c=='\r'||c=='\n'){ if(c=='\n') line++; pos++; continue; }
            if(c=='/' && src[pos+1]=='/'){ pos+=2; while(src[pos] && src[pos]!='\n') pos++; continue; }
            if(c=='/' && src[pos+1]=='*'){ pos+=2; while(src[pos] && !(src[pos]=='*'&&src[pos+1]=='/')) pos++; if(src[pos]) pos+=2; continue; }
            break;
        }
    }

    TTok number(){
        TTok t; t.t=TT_NUM; t.ival=0; int i=0;
        // Support hex numbers (0x prefix)
        if(src[pos] == '0' && (src[pos+1] == 'x' || src[pos+1] == 'X')) {
            pos += 2;
            t.v[i++] = '0'; t.v[i++] = 'x';
            while(i < 63 && ((src[pos] >= '0' && src[pos] <= '9') ||
                             (src[pos] >= 'a' && src[pos] <= 'f') ||
                             (src[pos] >= 'A' && src[pos] <= 'F'))) {
                char c = src[pos];
                t.v[i++] = c;
                if(c >= '0' && c <= '9') t.ival = t.ival * 16 + (c - '0');
                else if(c >= 'a' && c <= 'f') t.ival = t.ival * 16 + (c - 'a' + 10);
                else if(c >= 'A' && c <= 'F') t.ival = t.ival * 16 + (c - 'A' + 10);
                pos++;
            }
        } else {
            while(tcc_is_digit(src[pos])){ t.v[i++]=src[pos]; t.ival = t.ival*10 + (src[pos]-'0'); pos++; if(i>=63) break; }
        }
        t.v[i]=0; return t;
    }

    TTok ident(){
        TTok t; t.t=TT_ID; int i=0;
        while(tcc_is_alnum(src[pos])){ t.v[i++]=src[pos++]; if(i>=63) break; } t.v[i]=0;
        // Enhanced keywords with hardware and MMIO functions
        const char* kw[]={"int","char","string","return","if","else","while","break","continue",
                          "cin","cout","endl","argc","argv","read_file","write_file","append_file",
                          "array_size","array_resize","str_length","str_substr","int_to_str","str_compare",
                          "str_find_char","str_find_str","str_find_last_char","str_contains",
                          "str_starts_with","str_ends_with","str_count_char","str_replace_char",
                          "scan_hardware","get_device_info","get_hardware_array","display_memory_map",
                          "mmio_read8","mmio_read16","mmio_read32","mmio_read64",
                          "mmio_write8","mmio_write16","mmio_write32","mmio_write64",0};
        for(int k=0; kw[k]; ++k){ if(simple_strcmp(t.v,kw[k])==0){ t.t=TT_KW; break; } }
        return t;
    }

    TTok string(){
        TTok t; t.t=TT_STR; int i=0; pos++;
        while(src[pos] && src[pos]!='"'){ if(i<256) t.v[i++]=src[pos]; pos++; }
        t.v[i]=0; if(src[pos]=='"') pos++; return t;
    }

    TTok chlit(){
        TTok t; t.t=TT_CH; t.v[0]=0; int v=0; pos++; // skip '
        if(src[pos] && src[pos+1]=='\''){ v = (unsigned char)src[pos]; pos+=2; }
        t.ival = v; return t;
    }

    TTok op_or_punc(){
        TTok t; t.t=TT_OP; t.v[0]=src[pos]; t.v[1]=0; char c=src[pos];
        if(c=='<' && src[pos+1]=='<'){ t.v[0]='<'; t.v[1]='<'; t.v[2]=0; pos+=2; return t; }
        if(c=='>' && src[pos+1]=='>'){ t.v[0]='>'; t.v[1]='>'; t.v[2]=0; pos+=2; return t; }
        if((c=='='||c=='!'||c=='<'||c=='>') && src[pos+1]=='='){ t.v[0]=c; t.v[1]='='; t.v[2]=0; pos+=2; return t; }
        pos++; if(c=='('||c==')'||c=='{'||c=='}'||c==';'||c==','||c=='['||c==']') t.t=TT_PUNC; return t;
    }

    TTok next(){
        skipws();
        if(src[pos]==0){ TTok t; t.t=TT_EOF; t.v[0]=0; return t; }
        if(src[pos]=='"') return string();
        if(src[pos]=='\'') return chlit();
        if(tcc_is_digit(src[pos]) || (src[pos]=='0' && (src[pos+1]=='x'||src[pos+1]=='X'))) return number();
        if(tcc_is_alpha(src[pos])) return ident();
        return op_or_punc();
    }
};

// ============================================================
// Enhanced Parser / Compiler with Hardware and MMIO support
// ============================================================
struct TCompiler {
    TLex lx; TTok tk; TProgram pr;

    int brk_pos[32]; int brk_cnt=0;
    int cont_pos[32]; int cont_cnt=0;

    void adv(){ tk = lx.next(); }
    int  accept(const char* s){ if(simple_strcmp(tk.v,s)==0){ adv(); return 1; } return 0; }
    void expect(const char* s){ if(!accept(s)) { printf("Parse error near: %s\n", tk.v); } }

    void parse_primary(){
        if(tk.t==TT_NUM){ pr.emit1(T_PUSH_IMM); pr.emit4(tk.ival); adv(); return; }
        if(tk.t==TT_CH){ pr.emit1(T_PUSH_IMM); pr.emit4(tk.ival); adv(); return; }
        if(tk.t==TT_STR){ const char* p=pr.add_lit(tk.v); pr.emit1(T_PUSH_STR); pr.emit4((int)p); adv(); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"argc")==0){ pr.emit1(T_PUSH_ARGC); adv(); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"argv")==0){ adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_PUSH_ARGV_PTR); return; }

        // File I/O built-ins
        if(tk.t==TT_KW && simple_strcmp(tk.v,"read_file")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_READ_FILE); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"write_file")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_WRITE_FILE); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"append_file")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_APPEND_FILE); return;
        }

        // Array built-ins
        if(tk.t==TT_KW && simple_strcmp(tk.v,"array_size")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_ARRAY_SIZE); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"array_resize")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_ARRAY_RESIZE); return;
        }

        // String built-ins
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_length")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_STR_LENGTH); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_substr")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(",");
            parse_expression(); expect(")"); pr.emit1(T_STR_SUBSTR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"int_to_str")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_INT_TO_STR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_compare")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_COMPARE); return;
        }

        // String search functions
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_find_char")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_FIND_CHAR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_find_str")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_FIND_STR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_find_last_char")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_FIND_LAST_CHAR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_contains")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_CONTAINS); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_starts_with")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_STARTS_WITH); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_ends_with")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_ENDS_WITH); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_count_char")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_STR_COUNT_CHAR); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"str_replace_char")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(",");
            parse_expression(); expect(")"); pr.emit1(T_STR_REPLACE_CHAR); return;
        }

        // NEW: Hardware Discovery Functions
        if(tk.t==TT_KW && simple_strcmp(tk.v,"scan_hardware")==0){
            adv(); expect("("); expect(")"); pr.emit1(T_SCAN_HARDWARE); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"get_device_info")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_GET_DEVICE_INFO); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"get_hardware_array")==0){
            adv(); expect("("); expect(")"); pr.emit1(T_GET_HARDWARE_ARRAY); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"display_memory_map")==0){
            adv(); expect("("); expect(")"); pr.emit1(T_DISPLAY_MEMORY_MAP); return;
        }

        // NEW: Memory-Mapped I/O Functions
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_read8")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_MMIO_READ8); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_read16")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_MMIO_READ16); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_read32")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_MMIO_READ32); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_read64")==0){
            adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_MMIO_READ64); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_write8")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_MMIO_WRITE8); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_write16")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_MMIO_WRITE16); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_write32")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(")"); pr.emit1(T_MMIO_WRITE32); return;
        }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"mmio_write64")==0){
            adv(); expect("("); parse_expression(); expect(","); parse_expression(); expect(",");
            parse_expression(); expect(")"); pr.emit1(T_MMIO_WRITE64); return;
        }

        if(tk.t==TT_PUNC && tk.v[0]=='('){ adv(); parse_expression(); expect(")"); return; }

        if(tk.t==TT_ID){
            int idx = pr.get_local(tk.v);
            if(idx<0){ printf("Unknown var %s\n", tk.v); }
            char var_name[32]; simple_strcpy(var_name, tk.v);
            adv();

            // Array indexing
            if(tk.t==TT_PUNC && tk.v[0]=='['){
                pr.emit1(T_LOAD_LOCAL); pr.emit4(idx); // push handle
                adv(); // past '['
                parse_expression(); // push index
                expect("]");
                pr.emit1(T_LOAD_ARRAY);
                return;
            }

            pr.emit1(T_LOAD_LOCAL); pr.emit4(idx);
            return;
        }
    }

    void parse_unary(){
        if(accept("-")){ parse_unary(); pr.emit1(T_NEG); return; }
        parse_primary();
    }

    void parse_term(){
        parse_unary();
        while(tk.v[0]=='*' || tk.v[0]=='/'){
            char op=tk.v[0]; adv(); parse_unary();
            pr.emit1(op=='*'?T_MUL:T_DIV);
        }
    }

    void parse_arith(){
        parse_term();
        while(tk.v[0]=='+' || tk.v[0]=='-'){
            char op=tk.v[0]; adv(); parse_term();
            if(op=='+') {
                pr.emit1(T_ADD); // This will be overridden for strings in VM
            } else {
                pr.emit1(T_SUB);
            }
        }
    }

    void parse_cmp(){
        parse_arith();
        while(tk.t==TT_OP && (simple_strcmp(tk.v,"==")==0 || simple_strcmp(tk.v,"!=")==0 ||
              simple_strcmp(tk.v,"<")==0 || simple_strcmp(tk.v,"<=")==0 ||
              simple_strcmp(tk.v,">")==0 || simple_strcmp(tk.v,">=")==0)){
            char opv[3]; simple_strcpy(opv, tk.v); adv(); parse_arith();
            if(simple_strcmp(opv,"==")==0) pr.emit1(T_EQ);
            else if(simple_strcmp(opv,"!=")==0) pr.emit1(T_NE);
            else if(simple_strcmp(opv,"<")==0)  pr.emit1(T_LT);
            else if(simple_strcmp(opv,"<=")==0) pr.emit1(T_LE);
            else if(simple_strcmp(opv,">")==0)  pr.emit1(T_GT);
            else pr.emit1(T_GE);
        }
    }

    void parse_expression(){ parse_cmp(); }

    void parse_decl(unsigned char tkind){
        adv(); // past type keyword
        if(tk.t!=TT_ID){ printf("Expected identifier\n"); return; }
        char nm[32]; simple_strcpy(nm, tk.v); adv();

        int array_size = 0;
        // Array declaration syntax: int arr[size] or string arr[size]
        if(tk.t==TT_PUNC && tk.v[0]=='['){
            adv();
            if(tk.t==TT_NUM){
                array_size = tk.ival;
                adv();
            } else {
                printf("Expected array size\n"); return;
            }
            expect("]");

            if (tkind == 0) tkind = 3; // int -> int_array
            else if (tkind == 2) tkind = 4; // string -> string_array
        }

        int idx = pr.add_local(nm, tkind, array_size);

        // If it's an array, allocate it now, before parsing initializer
        if (tkind == 3 || tkind == 4) {
            pr.emit1(T_PUSH_IMM); pr.emit4(array_size);
            pr.emit1(T_ALLOC_ARRAY);
            pr.emit1(T_STORE_LOCAL); pr.emit4(idx);
        }

        if(accept("=")){
            if(tkind==3 || tkind==4){ // Array initialization
                expect("{");
                int i = 0;
                do {
                    if (tk.t == TT_PUNC && tk.v[0] == '}') break; // empty list or trailing comma
                    if (i >= array_size) {
                        printf("Too many initializers for array\n");
                        while(!accept("}")) { if(tk.t==TT_EOF) break; adv(); }
                        goto end_init;
                    }

                    pr.emit1(T_LOAD_LOCAL); pr.emit4(idx);      // 1. Push handle
                    pr.emit1(T_PUSH_IMM); pr.emit4(i);        // 2. Push index
                    parse_expression();                       // 3. Push value
                    pr.emit1(T_STORE_ARRAY);                    // 4. Store
                    i++;
                } while(accept(","));
                expect("}");
                end_init:;
            } else if(tkind==2){ // string
                if(tk.t==TT_STR){ const char* p=pr.add_lit(tk.v); pr.emit1(T_PUSH_STR); pr.emit4((int)p); adv(); }
                else if(tk.t==TT_KW && simple_strcmp(tk.v,"argv")==0){ adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_PUSH_ARGV_PTR); }
                else if(tk.t==TT_ID){ int j=pr.get_local(tk.v); adv(); pr.emit1(T_LOAD_LOCAL); pr.emit4(j); }
                else { parse_expression(); }
                pr.emit1(T_STORE_LOCAL); pr.emit4(idx);
            } else {
                parse_expression();
                pr.emit1(T_STORE_LOCAL); pr.emit4(idx);
            }
        }
        expect(";");
    }

    void parse_assign_or_coutcin(){
        if(tk.t==TT_KW && simple_strcmp(tk.v,"cout")==0){ adv();
            for(;;){
                expect("<<");
                if(tk.t==TT_KW && simple_strcmp(tk.v,"endl")==0){ adv(); pr.emit1(T_PRINT_ENDL); }
                else if(tk.t==TT_STR){ const char* p=pr.add_lit(tk.v); pr.emit1(T_PUSH_STR); pr.emit4((int)p); adv(); pr.emit1(T_PRINT_STR); }
                else if(tk.t==TT_KW && simple_strcmp(tk.v,"argv")==0){ adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_PUSH_ARGV_PTR); pr.emit1(T_PRINT_STR); }
                else if(tk.t==TT_ID){
                    char var_name[32]; simple_strcpy(var_name, tk.v);
                    int idx = pr.get_local(tk.v); int ty = pr.get_local_type(idx); adv();

                    // Handle array element printing vs whole array printing
                    if(tk.t==TT_PUNC && tk.v[0]=='['){
                        pr.emit1(T_LOAD_LOCAL); pr.emit4(idx); // load array
                        adv(); // past '['
                        parse_expression(); // push index
                        expect("]");
                        pr.emit1(T_LOAD_ARRAY); // load element
                        if (ty == 3) pr.emit1(T_PRINT_INT);      // int array element
                        else if (ty == 4) pr.emit1(T_PRINT_STR);  // string array element
                        else if (ty == 5) pr.emit1(T_PRINT_INT);  // device array element
                    } else {
                        pr.emit1(T_LOAD_LOCAL); pr.emit4(idx);
                        if(ty==4) pr.emit1(T_PRINT_STRING_ARRAY); // Print whole string array
                        else if(ty==3) pr.emit1(T_PRINT_INT_ARRAY); // Print whole int array
                        else if(ty==2) pr.emit1(T_PRINT_STR);
                        else if(ty==1) pr.emit1(T_PRINT_CHAR);
                        else pr.emit1(T_PRINT_INT);
                    }
                } else { parse_expression(); pr.emit1(T_PRINT_INT); }
                if(tk.t==TT_PUNC && tk.v[0]==';'){ adv(); break; }
            }
            return;
        }
        if (tk.t==TT_KW && simple_strcmp(tk.v,"cin")==0) {
			adv();
			for (;;) {
				expect(">>");
				if (tk.t != TT_ID) {
					printf("cin expects identifier\n");
					return;
				}
				int idx = pr.get_local(tk.v);
				int ty  = pr.get_local_type(idx);
				adv(); // past identifier

				// CRITICAL FIX: Emit the variable index with the READ instruction
				// so the VM knows WHERE to store the result
				if (ty == 2) {
					pr.emit1(T_READ_STR);
					pr.emit4(idx);  // Index to store into
				}
				else if (ty == 1) {
					pr.emit1(T_READ_CHAR);
					pr.emit4(idx);
				}
				else {
					pr.emit1(T_READ_INT);
					pr.emit4(idx);
				}

				// DO NOT emit T_STORE_LOCAL - the READ instruction handles storage

				if (tk.t == TT_PUNC && tk.v[0] == ';') {
					adv();
					break;
				}
			}
			return;
		}

        if(tk.t==TT_ID){
            int idx = pr.get_local(tk.v);
            if(idx<0){ printf("Unknown var %s\n", tk.v); }
            int ty = pr.get_local_type(idx);
            adv();

            // Array element assignment
            if(tk.t==TT_PUNC && tk.v[0]=='['){
                pr.emit1(T_LOAD_LOCAL); pr.emit4(idx);  // 1. Push handle
                adv(); // past '['
                parse_expression();                      // 2. Push index
                expect("]");
                expect("=");
                parse_expression();                      // 3. Push value
                pr.emit1(T_STORE_ARRAY);                    // 4. Store
                expect(";");
                return;
            }

            expect("=");
            if(ty==2){
                if(tk.t==TT_STR){ const char* p=pr.add_lit(tk.v); pr.emit1(T_PUSH_STR); pr.emit4((int)p); adv(); }
                else if(tk.t==TT_KW && simple_strcmp(tk.v,"argv")==0){ adv(); expect("("); parse_expression(); expect(")"); pr.emit1(T_PUSH_ARGV_PTR); }
                else if(tk.t==TT_ID){ int j=pr.get_local(tk.v); adv(); pr.emit1(T_LOAD_LOCAL); pr.emit4(j); }
                else { parse_expression(); }
            } else {
                parse_expression();
            }
            pr.emit1(T_STORE_LOCAL); pr.emit4(idx);
            expect(";");
            return;
        }

        // Expression statement
        parse_expression();
        pr.emit1(T_POP); // Pop unused result
        expect(";");
    }

    void parse_if(){
        adv(); expect("("); parse_expression(); expect(")");
        pr.emit1(T_JZ); int jz_at = pr.mark(); pr.emit4(0);
        parse_block();
        int has_else = (tk.t==TT_KW && simple_strcmp(tk.v,"else")==0);
        if(has_else){
            pr.emit1(T_JMP); int j_at = pr.mark(); pr.emit4(0);
            int here = pr.pc; pr.patch4(jz_at, here);
            adv(); // else
            parse_block();
            int end = pr.pc; pr.patch4(j_at, end);
        } else {
            int here = pr.pc; pr.patch4(jz_at, here);
        }
    }

    void parse_while(){
        adv(); expect("("); int cond_ip = pr.pc; parse_expression(); expect(")");
        pr.emit1(T_JZ); int jz_at = pr.mark(); pr.emit4(0);
        int brk_base=brk_cnt, cont_base=cont_cnt;
        parse_block();
        for(int i=cont_base;i<cont_cnt;i++){ pr.patch4(cont_pos[i], cond_ip); }
        cont_cnt = cont_base;
        pr.emit1(T_JMP); pr.emit4(cond_ip);
        int end_ip = pr.pc; pr.patch4(jz_at, end_ip);
        for(int i=brk_base;i<brk_cnt;i++){ pr.patch4(brk_pos[i], end_ip); }
        brk_cnt = brk_base;
    }

    void parse_block(){
        if(accept("{")){
            while(!(tk.t==TT_PUNC && tk.v[0]=='}') && tk.t!=TT_EOF) parse_stmt();
            expect("}");
        } else {
            parse_stmt();
        }
    }

    void parse_stmt(){
        if(tk.t==TT_KW && simple_strcmp(tk.v,"int")==0){ parse_decl(0); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"char")==0){ parse_decl(1); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"string")==0){ parse_decl(2); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"return")==0){ adv(); parse_expression(); pr.emit1(T_RET); expect(";"); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"if")==0){ parse_if(); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"while")==0){ parse_while(); return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"break")==0){ adv(); expect(";"); pr.emit1(T_JMP); int at=pr.mark(); pr.emit4(0); brk_pos[brk_cnt++]=at; return; }
        if(tk.t==TT_KW && simple_strcmp(tk.v,"continue")==0){ adv(); expect(";"); pr.emit1(T_JMP); int at=pr.mark(); pr.emit4(0); cont_pos[cont_cnt++]=at; return; }
        parse_assign_or_coutcin();
    }

    int compile(const char* source){
        lx.init(source); adv();
        if(!(tk.t==TT_KW && simple_strcmp(tk.v,"int")==0)) { printf("Expected 'int' at start\n"); return -1; }
        adv();
        if(!(tk.t==TT_ID && simple_strcmp(tk.v,"main")==0)){ printf("Expected main\n"); return -1; }
        adv(); expect("("); expect(")"); parse_block();
        pr.emit1(T_PUSH_IMM); pr.emit4(0); pr.emit1(T_RET);
        return pr.pc;
    }
};

// ============================================================
// MODIFIED TinyVM FOR PARTIAL COMPUTING
// ============================================================
struct TinyVM {
    static const int STK_MAX = 1024;
    int   stk[STK_MAX]; int sp=0;
    int   locals[TProgram::LOC_MAX];
    int   argc; const char** argv;
    TProgram* P;
    char str_in[256];
    uint64_t ahci_base; int port; // for file I/O
    int bound_window_idx = -1; 
	int pending_store_idx = 0;  // Where to store READ result

    // --- EXECUTION STATE FOR PARTIAL COMPUTING ---
    int ip = 0;             // Instruction Pointer (Persistent)
    bool is_running = false; // Is a program currently active?
    int exit_code = 0;      // Store result when finished
    // ---------------------------------------------
	// --- NEW: ASYNC INPUT STATE ---
	Window* bound_window = nullptr;   // instead of int bound_window_idx
    bool waiting_for_input = false;
    int  input_mode = 0;
    char input_buffer[256];
    int  input_pos = 0;
    // String pool for dynamic string management
    static const int STRING_POOL_SIZE = 8192;
    char string_pool[STRING_POOL_SIZE];
    int string_pool_top = 0;

    // Simple array management
    struct Array {
        int* data;
        int size;
        int capacity;
    };
    static const int MAX_ARRAYS = 64;
    Array arrays[MAX_ARRAYS];
    int array_count = 0;

    // Special array handle for hardware devices
    int hardware_array_handle = 0;

    inline void push(int v){ if(sp<STK_MAX) stk[sp++]=v; }
    inline int  pop(){ return sp?stk[--sp]:0; }

    // --- Helper Methods (Same as before) ---
    uint8_t mmio_read_8(uint64_t addr) { return *(volatile uint8_t*)addr; }
    uint16_t mmio_read_16(uint64_t addr) { return *(volatile uint16_t*)addr; }
    uint32_t mmio_read_32(uint64_t addr) { return *(volatile uint32_t*)addr; }
    uint64_t mmio_read_64(uint64_t addr) { return *(volatile uint64_t*)addr; }
    bool mmio_write_8(uint64_t addr, uint8_t value) { *(volatile uint8_t*)addr = value; return true; }
    bool mmio_write_16(uint64_t addr, uint16_t value) { *(volatile uint16_t*)addr = value; return true; }
    bool mmio_write_32(uint64_t addr, uint32_t value) { *(volatile uint32_t*)addr = value; return true; }
    bool mmio_write_64(uint64_t addr, uint64_t value) { *(volatile uint64_t*)addr = value; return true; }

    // String helpers
    const char* alloc_string(int len) {
        if(string_pool_top + len + 1 > STRING_POOL_SIZE) string_pool_top = 0; 
        if(string_pool_top + len + 1 > STRING_POOL_SIZE) return nullptr;
        char* result = &string_pool[string_pool_top];
        string_pool_top += len + 1;
        return result;
    }
    // (Simplified string helpers for brevity - assuming originals exist or use minimal versions)
    // Note: Ensure your original string helper functions (concat_strings, etc.) are inside here or available.
	  
    // Array helpers
    int alloc_array(int size) {
        if(array_count >= MAX_ARRAYS) return 0;
        int handle = array_count + 1;
        arrays[array_count].data = new int[size];
        arrays[array_count].size = size;
        arrays[array_count].capacity = size;
        for(int i=0; i<size; i++) arrays[array_count].data[i] = 0;
        array_count++;
        return handle;
    }
    Array* get_array(int handle) {
        if(handle > 0 && handle <= array_count) return &arrays[handle-1];
        return nullptr;
    }
    int resize_array(int handle, int new_size) {
        Array* arr = get_array(handle);
        if(!arr) return 0;
        int* new_data = new int[new_size];
        int copy_size = (arr->size < new_size) ? arr->size : new_size;
        for(int i=0; i<copy_size; i++) new_data[i] = arr->data[i];
        for(int i=copy_size; i<new_size; i++) new_data[i] = 0;
        delete[] arr->data;
        arr->data = new_data;
        arr->size = new_size;
        arr->capacity = new_size;
        return handle;
    }
    int create_device_info_array(int index) { return 0; /* stub */ }
    int create_hardware_array() { return 0; /* stub */ }
    int scan_hardware() { return hardware_count; }

    // --- NEW: INITIALIZE EXECUTION ---
     // UPDATED start_execution
      void start_execution(TProgram& prog,
                         int ac,
                         const char** av,
                         uint64_t base,
                         int p,
                         Window* win)   // NOTE: Window* not int
    {
        bound_window = win;
        P=&prog; argc=ac; argv=av; ahci_base=base; port=p;
        sp=0; ip=0; is_running=true; exit_code=0;
        waiting_for_input = false; input_mode = 0; input_pos = 0;
        array_count = 0; hardware_array_handle = 0; string_pool_top = 0;
        for (int i=0;i<TProgram::LOC_MAX;i++) locals[i]=0;

        for(int i = 0; i < P->loc_count; i++) {
            if(P->loc_type[i] == 3 || P->loc_type[i] == 4) {
                int arr_handle = alloc_array(P->loc_array_size[i]);
                locals[i] = arr_handle;
            }
        }
    }
	
	void vm_print(const char* s) {
		if (bound_window) {
			bound_window->console_print(s);  // Route to window!
		} else {
			printf(s);  // Fallback if no window
		}
	}

	void vm_putc(char c) {
		if (bound_window) {
			// Use put_char directly, which adds to current line without wrapping
			bound_window->put_char(c);
		} else {
			printf("%c", c);  // Fallback
		}
	}

    void feed_input(char c) {
		if (!waiting_for_input) return;
		
		if (c == '\n' || c == '\r') {
			input_buffer[input_pos] = 0;
			vm_putc('\n');
			
			if (input_mode == 1) {
				locals[pending_store_idx] = simple_atoi(input_buffer);
			}
			else if (input_mode == 2) {
				locals[pending_store_idx] = (unsigned char)input_buffer[0];
			}
			else if (input_mode == 3) {
				simple_strcpy(str_in, input_buffer);
				locals[pending_store_idx] = (int)str_in;
			}
			
			waiting_for_input = false;
			input_pos = 0;
			input_mode = 0;
			pending_store_idx = 0;
			
		} else if (c == '\b') {
			if (input_pos > 0) {
				input_pos--;
				input_buffer[input_pos] = 0;
				vm_putc('\b');
			}
		} else if (c >= 32 && c <= 126 && input_pos < 255) {
			input_buffer[input_pos++] = c;
			vm_putc(c);
		}
	}



    // --- NEW: TICK FUNCTION (Runs 'steps' instructions) ---
    // Returns: 1 if still running, 0 if finished
    int tick(int steps) {
        if (!is_running) return 0;
        if (waiting_for_input) return 1; // Still running, just paused

        int steps_done = 0;
        while(steps_done < steps && ip < P->pc && is_running){
			if (waiting_for_input) break;

            TOp op = (TOp)P->code[ip++];

            // PASTE YOUR ORIGINAL HUGE SWITCH STATEMENT HERE
            // IMPORTANT MODIFICATION: Replace "return rv;" with "exit_code = rv; is_running=false; return 0;"
            switch(op){
                case T_NOP: break;
                case T_PUSH_IMM: { int v= *(int*)&P->code[ip]; ip+=4; push(v); } break;
                case T_PUSH_STR: { int p= *(int*)&P->code[ip]; ip+=4; push(p); } break;
                case T_LOAD_LOCAL:{ int i=*(int*)&P->code[ip]; ip+=4; push(locals[i]); } break;
                case T_STORE_LOCAL:{ int i=*(int*)&P->code[ip]; ip+=4; locals[i]=pop(); } break;
                case T_POP: { if(sp) --sp; } break;
                case T_ADD: { int b=pop(), a=pop(); push(a+b); } break;
                case T_SUB: { int b=pop(), a=pop(); push(a-b); } break;
                case T_MUL: { int b=pop(), a=pop(); push(a*b); } break;
                case T_DIV: { int b=pop(), a=pop(); push(b? a/b:0); } break;
                case T_NEG: { int a=pop(); push(-a); } break;
                case T_EQ: { int b=pop(), a=pop(); push(a==b); } break;
                case T_NE: { int b=pop(), a=pop(); push(a!=b); } break;
                case T_LT: { int b=pop(), a=pop(); push(a<b); } break;
                case T_GT: { int b=pop(), a=pop(); push(a>b); } break;
                case T_LE: { int b=pop(), a=pop(); push(a<=b); } break;
                case T_GE: { int b=pop(), a=pop(); push(a>=b); } break;
                case T_JMP: { int t=*(int*)&P->code[ip]; ip=t; } break;
                case T_JZ:  { int t=*(int*)&P->code[ip]; ip+=4; int v=pop(); if(v==0) ip=t; } break;
                case T_JNZ: { int t=*(int*)&P->code[ip]; ip+=4; int v=pop(); if(v!=0) ip=t; } break;
                case T_PRINT_INT: {
					int v = pop();
					char buf[16];
					int_to_string(v, buf);
					printf("%s", buf);
				} break;

				case T_PRINT_CHAR: {
					int v = pop();
					char buf[2] = { (char)(v & 0xFF), 0 };
					printf("%s", buf);
				} break;

				case T_PRINT_STR: {
					const char* p = (const char*)pop();
					if (p) printf("%s", p);
				} break;

				case T_PRINT_ENDL: {
					printf("\n");
				} break;

				case T_PRINT_INT_ARRAY: {
					int handle = pop();
					Array* arr = get_array(handle);
					if (arr) {
						for (int i = 0; i < arr->size; i++) {
							char buf[16];
							int_to_string(arr->data[i], buf);
							printf("%s", buf);
							if (i < arr->size - 1) printf(", ");
						}
					}
				} break;

				case T_PRINT_STRING_ARRAY: {
					int handle = pop();
					// Handle string array printing
				} break;

				case T_READ_INT: {
					int idx = *(int*)&P->code[ip]; ip+=4;  // READ the variable index
					waiting_for_input = true;
					input_mode = 1;
					input_pos = 0;
					pending_store_idx = idx;  // Store where to write the result
					return 1;
				} break;

				case T_READ_CHAR: {
					int idx = *(int*)&P->code[ip]; ip+=4;
					waiting_for_input = true;
					input_mode = 2;
					input_pos = 0;
					pending_store_idx = idx;
					return 1;
				} break;

				case T_READ_STR: {
					int idx = *(int*)&P->code[ip]; ip+=4;
					waiting_for_input = true;
					input_mode = 3;
					input_pos = 0;
					pending_store_idx = idx;
					return 1;
				} break;
                // CRITICAL CHANGE: RETURN HANDLING
                case T_RET: { 
                    int rv=pop(); 
                    exit_code = rv; 
                    is_running = false; 
                    return 0; // Finished
                } break;
                
                default: break;
            }
            steps_done++;
        }
        
        if (ip >= P->pc && !waiting_for_input) {
            is_running = false;
            return 0;
        }

        return 1; // Still running
    }
};
// --- GLOBAL VM STATE ---

// --- GLOBAL PROCESS TABLE ---
#define MAX_PROCESSES 4
/* processes[] and prog_pool[] removed — RunContext/ExecContext own their
   own TinyVM+TProgram instances; these globals were dead weight (~400 KB). */

// ============================================================
// Enhanced Object I/O (TVM3 - with hardware support)
// ============================================================
struct TVMObject {
    static int save(uint64_t base, int port, const char* path, const TProgram& P){
        static unsigned char buf[ TProgram::CODE_MAX + TProgram::LIT_MAX + 128 ];
        int off=0;
        buf[off++]='T'; buf[off++]='V'; buf[off++]='M'; buf[off++]='3'; // Version 3 with hardware support
        *(int*)&buf[off]=P.pc; off+=4;
        *(int*)&buf[off]=P.lit_top; off+=4;
        *(int*)&buf[off]=P.loc_count; off+=4;
        simple_memcpy(&buf[off], P.code, P.pc); off+=P.pc;
        simple_memcpy(&buf[off], P.lit, P.lit_top); off+=P.lit_top;

        // Save local variable metadata (names, types, array sizes)
        for(int i = 0; i < P.loc_count; i++) {
            int name_len = tcc_strlen(P.loc_name[i]) + 1;
            simple_memcpy(&buf[off], P.loc_name[i], name_len); off += name_len;
            buf[off++] = P.loc_type[i];
            *(int*)&buf[off] = P.loc_array_size[i]; off += 4;
        }

        return fat32_write_file(path, buf, off);
    }

    // In SECTION 6, inside the TVMObject struct

static int load(uint64_t base, int port, const char* path, TProgram& P){
    // FIX: First, get the file's directory entry to find its true size.
    fat_dir_entry_t entry;
    uint32_t sector, offset;
    if (fat32_find_entry(path, &entry, &sector, &offset) != 0) {
        return -1; // File not found
    }
    uint32_t n = entry.file_size; // Use the REAL size from the filesystem.

    // Now we can read the file content.
    char* buf = fat32_read_file_as_string(path);
    if (!buf) {
        return -1; // Read failed
    }

    // The original buggy line is no longer needed.
    // int n = tcc_strlen(buf);  

    if (n < 16) { 
        delete[] buf; 
        return -1; 
    }
    if (!(buf[0] == 'T' && buf[1] == 'V' && buf[2] == 'M' && (buf[3] == '1' || buf[3] == '2' || buf[3] == '3'))) {
        delete[] buf;
        return -2;
    }
    int cp = *(int*)&buf[4], lp = *(int*)&buf[8], lc = *(int*)&buf[12];
    if (cp < 0 || cp > TProgram::CODE_MAX || lp < 0 || lp > TProgram::LIT_MAX || lc < 0 || lc > TProgram::LOC_MAX) {
        delete[] buf;
        return -3;
    }

    // The rest of the function now works correctly because 'n' is the true file size.
    P.pc = cp; P.lit_top = lp; P.loc_count = lc;
    int off = 16;
    simple_memcpy(P.code, &buf[off], cp); off += cp;
    simple_memcpy(P.lit, &buf[off], lp); off += lp;

    if (buf[3] >= '2') {
        for (int i = 0; i < lc; i++) {
            int name_len = 0;
            while (off + name_len < n && buf[off + name_len] != 0) name_len++;
            
            if (name_len < 32) {
                simple_memcpy(P.loc_name[i], &buf[off], name_len + 1);
            } else {
                P.loc_name[i][0] = 0;
            }
            off += name_len + 1;
            
            // Boundary check before reading type and size
            if (off + 5 > n) {
                delete[] buf;
                return -4; // Corrupt file, not enough data for metadata
            }
            
            P.loc_type[i] = buf[off++];
            P.loc_array_size[i] = *(int*)&buf[off]; off += 4;
        }
    } else {
        for (int i = 0; i < lc; i++) {
            P.loc_name[i][0] = 0;
            P.loc_type[i] = 0;
            P.loc_array_size[i] = 0;
        }
    }
    delete[] buf;
    return 0;
};
};

// ============================================================
// Enhanced compile/run entry points
// ============================================================
static int tinyvm_compile_to_obj(uint64_t ahci_base, int port, const char* src_path, const char* obj_path){
    char* srcbuf = fat32_read_file_as_string(src_path);
    if(!srcbuf){ printf("read fail\n"); return -1; }
    TCompiler C; int ok = C.compile(srcbuf);
    delete[] srcbuf;
    if(ok<0){ printf("Compilation failed!\n"); return -2; }
    int w = TVMObject::save(ahci_base, port, obj_path, C.pr);
    if(w<0){ printf("write fail\n"); return -3; }
    return 0;
}
// Updated wrapper (optional)
static int tinyvm_run_obj(uint64_t ahci_base, int port, const char* obj_path, int argc, const char** argv) {
    // Forward the actual parameters passed to this function
    cmd_run(ahci_base, port, obj_path);
    return 0;
}



// ============================================================
// Enhanced Shell glue with hardware discovery info
// ============================================================
extern "C" void cmd_compile(uint64_t ahci_base, int port, const char* filename){
    if (!filename) { printf("Usage: compile <file.cpp>\n"); return; }
    static char obj[64]; int i=0; while(filename[i] && i<60){ obj[i]=filename[i]; i++; }
    while(i>0 && obj[i-1] != '.') i--; obj[i]=0; simple_strcpy(&obj[i], "obj");
    printf("Compiling %s...\n", filename);
    int r = tinyvm_compile_to_obj(ahci_base, port, filename, obj);
    if(r==0) { printf("OK -> %s\n", obj); } else { printf("Compilation failed!\n"); }
}


// --- Command parsing helper ---
char* get_arg(char* args, int n) {
    char* p = args;

    // Loop to find the start of the Nth argument
    for (int i = 0; i < n; i++) {
        // Skip leading spaces for the current argument
        while (*p && *p == ' ') p++;

        // If we're at the end of the string, the requested arg doesn't exist
        if (*p == '\0') return nullptr;

        // Skip over the content of the current argument
        if (*p == '"') {
            p++; // Skip opening quote
            while (*p && *p != '"') p++;
            if (*p == '"') p++; // Skip closing quote
        } else {
            while (*p && *p != ' ') p++;
        }
    }

    // Now p is at the start of the Nth argument (or spaces before it)
    while (*p && *p == ' ') p++;
    if (*p == '\0') return nullptr;

    char* arg_start = p;
    if (*p == '"') {
        arg_start++; // The actual argument starts after the quote
        p++;
        while (*p && *p != '"') p++;
        if (*p == '"') *p = '\0'; // Place null terminator on the closing quote
    } else {
        while (*p && *p != ' ') p++;
        if (*p) *p = '\0'; // Place null terminator on the space
    }
    return arg_start;
}


    // Separate process tables
#define MAX_RUN_PROCESSES 4
#define MAX_EXEC_PROCESSES 4

// Context for RUN processes (disk-based execution)
struct RunContext {
    TProgram prog;           // Program code
    uint64_t ahci_base;      // Disk controller base
    int port;                // Disk port
    TinyVM vm;               // VM instance
    bool active;             // Is this slot in use
    char filename[64];       // Source filename for debugging
};

// Context for EXEC processes (memory-based execution)  
struct ExecContext {
    TProgram prog;           // Program code
    TinyVM vm;               // VM instance
    bool active;             // Is this slot in use
    int exec_id;             // Unique execution ID
};
static RunContext run_contexts[MAX_RUN_PROCESSES];
static ExecContext exec_contexts[MAX_EXEC_PROCESSES];
extern "C" void cmd_run(uint64_t ahci_base, int port, const char* filename) {
    if (!filename) { return; }
    // Find a free run slot
    for (int i = 0; i < MAX_RUN_PROCESSES; i++) {
        if (!run_contexts[i].active) {
            RunContext& ctx = run_contexts[i];
            ctx.active = false;
            ctx.ahci_base = ahci_base;
            ctx.port = port;
            strncpy(ctx.filename, filename, 63);
            ctx.filename[63] = '\0';
            if (TVMObject::load(ahci_base, port, filename, ctx.prog) != 0) {
                return;
            }
            const char* av[] = { filename, nullptr };
            ctx.vm.start_execution(ctx.prog, 1, av, ahci_base, port, nullptr);
            ctx.active = true;
            return;
        }
    }
}

	
	// List active run processes
void list_run_processes() {
    wm.print_to_focused("Active RUN processes:\n");
    bool found = false;
    for (int i = 0; i < MAX_RUN_PROCESSES; i++) {
        if (run_contexts[i].active) {
            char msg[128];
            snprintf(msg, 128, "  Slot %d: %s (IP=%d)\n", 
                     i, run_contexts[i].filename, run_contexts[i].vm.ip);
            wm.print_to_focused(msg);
            found = true;
        }
    }
    if (!found) {
        wm.print_to_focused("  (none)\n");
    }
}




// List active exec processes
void list_exec_processes() {
    wm.print_to_focused("Active EXEC processes:\n");
    bool found = false;
    for (int i = 0; i < MAX_EXEC_PROCESSES; i++) {
        if (exec_contexts[i].active) {
            char msg[128];
            snprintf(msg, 128, "  Slot %d: ID=%d (IP=%d)\n", 
                     i, exec_contexts[i].exec_id, exec_contexts[i].vm.ip);
            wm.print_to_focused(msg);
            found = true;
        }
    }
    if (!found) {
        wm.print_to_focused("  (none)\n");
    }
}

// Kill a run process
void kill_run_process(int slot) {
    if (slot >= 0 && slot < MAX_RUN_PROCESSES && run_contexts[slot].active) {
        run_contexts[slot].active = false;
        run_contexts[slot].vm.is_running = false;
        wm.print_to_focused("RUN process killed.\n");
    } else {
        wm.print_to_focused("Invalid RUN slot.\n");
    }
}

// Kill an exec process
void kill_exec_process(int slot) {
    if (slot >= 0 && slot < MAX_EXEC_PROCESSES && exec_contexts[slot].active) {
        exec_contexts[slot].active = false;
        exec_contexts[slot].vm.is_running = false;
        wm.print_to_focused("EXEC process killed.\n");
    } else {
        wm.print_to_focused("Invalid EXEC slot.\n");
    }
}
	
	

	
bool run_process_waiting_for_input() {
    for (int i = 0; i < MAX_RUN_PROCESSES; i++) {
        if (run_contexts[i].active && run_contexts[i].vm.waiting_for_input) {
            return true;
        }
    }
    return false;
}
		
	// =============================================================================
// ELF32 LOADER AND PROCESS EXECUTION
// =============================================================================

// ELF32 Header structures
#define EI_NIDENT 16
#define EI_MAG0 0
#define EI_MAG1 1
#define EI_MAG2 2
#define EI_MAG3 3
#define EI_CLASS 4
#define EI_DATA 5

#define ELFMAG0 0x7f
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'
#define ELFCLASS32 1
#define ELFDATA2LSB 1

#define ET_EXEC 2
#define EM_386 3

#define PT_LOAD 1
#define PF_X 1
#define PF_W 2
#define PF_R 4

typedef struct {
    uint8_t  e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} __attribute__((packed)) Elf32_Ehdr;

typedef struct {
    uint32_t p_type;
    uint32_t p_offset;
    uint32_t p_vaddr;
    uint32_t p_paddr;
    uint32_t p_filesz;
    uint32_t p_memsz;
    uint32_t p_flags;
    uint32_t p_align;
} __attribute__((packed)) Elf32_Phdr;

// =============================================================================
// TERMINAL WINDOW IMPLEMENTATION
// =============================================================================
static constexpr int TERM_HEIGHT = 35;
static constexpr int TERM_WIDTH  = 120;
char prompt_buffer[TERM_WIDTH];

// =============================================================================
// MATRIX ARRAY STORE + DESKTOP SUITE  (patch — see OS-main-patch/INTEGRATION.md)
// =============================================================================
#include "matrix_array.h"
#include "desktop_suite/launcher.h"

// parse "rwxt" → bitmask of NPA_R/W/RX/TX. Default: r+w+x if empty.
static uint16_t parse_perms(const char* s) {
    if (!s || !*s) return NPA_R | NPA_W | NPA_RX;
    uint16_t p = 0;
    for (; *s; ++s) {
        switch (*s) {
            case 'r': case 'R': p |= NPA_R;  break;
            case 'w': case 'W': p |= NPA_W;  break;
            case 'x': case 'X': p |= NPA_RX; break;
            case 't': case 'T': p |= NPA_TX; break;
            default: break;
        }
    }
    return p;
}

// NpaPrint adapter — body defined after TerminalWindow is complete.
void npa_term_print(void* ctx, const char* s);

class TerminalWindow : public Window {
private:
    // Terminal state
    char buffer[TERM_HEIGHT][TERM_WIDTH];
    int line_count;
    char current_line[TERM_WIDTH];
    int line_pos;

    // True when the last character pushed by console_print() was a '\n'
    // (or no output has been printed yet). When false, the next
    // console_print() call must CONTINUE the current buffer line rather
    // than starting a fresh one. This is what stops chatty guest output
    // — which arrives a few bytes at a time, one console_print() per
    // tick batch — from getting a spurious line break every few bytes.
    bool output_at_line_start = true;

    // Editor state
    bool in_editor;
    char edit_filename[32];
    char** edit_lines;
    int edit_line_count;
    int edit_current_line;
    int edit_cursor_col;
    int edit_scroll_offset;

    // Prompt visual state for multi-line input
    int prompt_visual_lines;
    char private_startup_cmd[256];
// Editor viewport settings
static constexpr int EDIT_ROWS = 35;       // rows visible in the editor area
static constexpr int EDIT_COL_PIX = 8;     // font width
static constexpr int EDIT_LINE_PIX = 10;   // line height
public:  // put_char overrides Window::put_char
void put_char(char c) override {
        if (in_editor) return; // Don't mess with editor

        // Ensure we have at least one line
        if (line_count == 0) {
            push_line("");
        }

        // Get the last line in the buffer
        char* line = buffer[line_count - 1];
        int len = strlen(line);

        if (c == '\n') {
            push_line(""); // Real newline
        } 
        else if (c == '\b') {
            if (len > 0) {
                line[len - 1] = 0; // Remove last char
            }
        } 
        else if (c >= 32 && c <= 126) {
            // Check if line is full
            if (len < TERM_WIDTH - 1) {
                line[len] = c;
                line[len + 1] = 0;
            } else {
                // Wrap to new line
                char temp[2] = {c, 0};
                push_line(temp);
            }
        }
    }
void editor_clamp_cursor_to_line() {
    if (edit_current_line < 0) edit_current_line = 0;
    if (edit_current_line >= edit_line_count) edit_current_line = edit_line_count - 1;
    if (edit_current_line < 0) edit_current_line = 0; // handle empty
    if (edit_line_count > 0) {
        int len = (int)strlen(edit_lines[edit_current_line]);
        if (edit_cursor_col > len) edit_cursor_col = len;
        if (edit_cursor_col < 0) edit_cursor_col = 0;
    } else {
        edit_cursor_col = 0;
    }
}

void editor_ensure_cursor_visible() {
    if (edit_current_line < edit_scroll_offset) {
        edit_scroll_offset = edit_current_line;
        if (edit_scroll_offset < 0) edit_scroll_offset = 0;
    } else if (edit_current_line >= edit_scroll_offset + EDIT_ROWS) {
        edit_scroll_offset = edit_current_line - (EDIT_ROWS - 1);
    }
}
private:
    // Insert a new line at a given index, copying the provided text into it.
    void editor_insert_line_at(int index, const char* text) {
        if (index < 0 || index > edit_line_count) return;

        char** new_lines = new char*[edit_line_count + 1];

        for (int i = 0; i < index; ++i) {
            new_lines[i] = edit_lines[i];
        }

        new_lines[index] = new char[TERM_WIDTH];
        memset(new_lines[index], 0, TERM_WIDTH);
        if (text) {
            strncpy(new_lines[index], text, TERM_WIDTH - 1);
        }

        for (int i = index; i < edit_line_count; ++i) {
            new_lines[i + 1] = edit_lines[i];
        }

        if (edit_lines) {
            delete[] edit_lines;
        }
        edit_lines = new_lines;
        edit_line_count++;
    }

    // Delete the line at a given index.
    void editor_delete_line_at(int index) {
        if (index < 0 || index >= edit_line_count || edit_line_count <= 1) return;

        delete[] edit_lines[index];

        char** new_lines = new char*[edit_line_count - 1];
        
        for (int i = 0; i < index; ++i) {
            new_lines[i] = edit_lines[i];
        }

        for (int i = index + 1; i < edit_line_count; ++i) {
            new_lines[i - 1] = edit_lines[i];
        }

        delete[] edit_lines;
        edit_lines = new_lines;
        edit_line_count--;
    }

    // Get visible columns for the first prompt line (accounts for "> ")
    int term_cols_first() const {
        int cols = (w - 10) / 8;
        cols -= 2;
        if (cols < 1) cols = 1;
        if (cols > 118) cols = 118;
        return cols;
    }

    // Get visible columns for continuation lines or general output
    int term_cols_cont() const {
        int cols = (w - 10) / 8;
        if (cols < 1) cols = 1;
        if (cols > 118) cols = 118;
        return cols;
    }

    // Removes the last N lines from the terminal buffer (used to refresh prompt)
    void remove_last_n_lines(int n) {
        while (n-- > 0 && line_count > 0) {
            memset(buffer[line_count - 1], 0, 120);
            line_count--;
        }
    }

    // Finds the best position to wrap a string within max_cols
    int find_wrap_pos(const char* s, int max_cols) {
        int len = (int)strlen(s);
        if (len <= max_cols) return len;

        int wrap_at = max_cols;
        for (int i = max_cols; i > 0; --i) {
            if (s[i] == ' ' || s[i] == '\t' || s[i] == '-') {
                wrap_at = i;
                break;
            }
        }
        return wrap_at;
    }

    // Pushes a single line segment of the prompt to the terminal buffer
    void append_prompt_line(const char* seg, bool first) {
        char linebuf[120];
        linebuf[0] = 0;
        if (first) {
            snprintf(linebuf, 120, "> %s", seg);
        } else {
            snprintf(linebuf, 120, "  %s", seg);
        }
        push_line(linebuf);
    }

    // Redraws the entire multi-line prompt based on `current_line`
    void update_prompt_display() {
        if (prompt_visual_lines > 0) {
            remove_last_n_lines(prompt_visual_lines);
            prompt_visual_lines = 0;
        }

        const char* p = current_line;
        bool first = true;
        int seg_count = 0;

        if (*p == '\0') {
            append_prompt_line("", true);
            prompt_visual_lines = 1;
            return;
        }

        while (*p) {
            int max_cols = first ? term_cols_first() : term_cols_cont();
            int take = find_wrap_pos(p, max_cols);

            char seg[120];
            strncpy(seg, p, take);
            seg[take] = '\0';
            
            int trim = (int)strlen(seg);
            while (trim > 0 && (seg[trim-1] == ' ' || seg[trim-1] == '\t')) {
                seg[--trim] = '\0';
            }

            append_prompt_line(seg, first);
            seg_count++;

            p += take;
            if (*p == ' ' || *p == '\t') p++;
            first = false;
        }
        prompt_visual_lines = seg_count;
    }

    // Append a text fragment to the LAST buffer line (no re-wrapping of
    // existing content, no newline). Used by push_wrapped_text to
    // continue a line that a previous console_print() call left
    // unterminated.
    //
    // `cols` is the window's CURRENT visible column count
    // (term_cols_cont()), not TERM_WIDTH. TERM_WIDTH is just the size of
    // the internal char buffer (120) — it has nothing to do with how
    // many characters actually fit on screen. draw_string() draws
    // glyphs with no clipping against the window border, so a
    // continuation line that's allowed to grow up to TERM_WIDTH-1 chars
    // (as it previously did) draws straight past the right edge of any
    // window narrower than ~119 columns. That was the source of the
    // "glitchy" streaking/overflow seen with chatty ELF guest output:
    // each chunk just kept appending to the same on-screen row instead
    // of wrapping at the column the window can actually display.
    void append_to_last_line(const char* frag, int cols) {
        if (!frag || !*frag) return;
        if (cols < 1) cols = 1;
        if (cols > TERM_WIDTH - 1) cols = TERM_WIDTH - 1;
        if (line_count == 0) push_line("");
        char* line = buffer[line_count - 1];
        int len = (int)strlen(line);
        while (*frag) {
            if (len >= cols) {
                push_line("");
                line = buffer[line_count - 1];
                len  = 0;
            }
            line[len++] = *frag++;
            line[len]   = '\0';
        }
    }

    // Pushes word-wrapped text (from console_print) to the buffer.
    //
    // A single logical output line is only ever terminated by an actual
    // '\n' in the input. Text that arrives WITHOUT a trailing newline
    // leaves the line "open": output_at_line_start is set false, and the
    // next call continues that same buffer line via append_to_last_line.
    // Previously every call unconditionally push_line()'d its text, so a
    // guest streaming output a few bytes per tick (one console_print per
    // batch) got a line break every few bytes.
    void push_wrapped_text(const char* s, int cols) {
        const char* p = s;
        while (*p) {
            const char* nl = strchr(p, '\n');
            bool has_newline = (nl != nullptr);
            if (!nl) nl = p + strlen(p);

            char line[512]; // Temporary buffer for a logical line
            int len = nl - p;
            if (len > 511) len = 511;
            strncpy(line, p, len);
            line[len] = '\0';

            if (len == 0) {
                // Empty segment. If it came from a real '\n' it is a
                // blank line — but only "blank" if the current line was
                // already started fresh; otherwise the '\n' just closes
                // the open line and emits nothing extra.
                if (has_newline && output_at_line_start) {
                    push_line("");
                }
            } else if (!output_at_line_start) {
                // Continue the open buffer line. We intentionally do NOT
                // re-wrap already-drawn content here; append_to_last_line
                // spills onto a fresh line once it hits the window's
                // actual visible width.
                append_to_last_line(line, cols);
            } else {
                // Fresh line: word-wrap as before.
                const char* q = line;
                while (*q) {
                    int take = find_wrap_pos(q, cols);
                    char seg[120];
                    strncpy(seg, q, take);
                    seg[take] = '\0';

                    int trim = (int)strlen(seg);
                    while (trim > 0 && (seg[trim-1] == ' ' || seg[trim-1] == '\t')) {
                        seg[--trim] = '\0';
                    }

                    push_line(seg);
                    q += take;
                    if (*q == ' ' || *q == '\t') q++;
                }
            }

            // The line is "closed" (next text starts fresh) only when we
            // actually consumed a '\n'. Otherwise it stays open so the
            // following console_print() continues it.
            output_at_line_start = has_newline;

            p = (*nl == '\n') ? nl + 1 : nl;
        }
    }

    // --- END OF MODULE ---

    void scroll() {
        memmove(buffer[0], buffer[1], (TERM_HEIGHT - 1) * TERM_WIDTH);
        memset(buffer[TERM_HEIGHT - 1], 0, TERM_WIDTH);
    }

    void push_line(const char* s) {
        if (line_count >= TERM_HEIGHT) {
            scroll();
            strncpy(buffer[TERM_HEIGHT - 1], s, TERM_WIDTH - 1);
        } else {
            strncpy(buffer[line_count], s, TERM_WIDTH - 1);
            line_count++;
        }
    }
    void print_prompt() { 
        snprintf(prompt_buffer, TERM_WIDTH, "> %s", current_line);
        if (line_count > 0) {
            strncpy(buffer[line_count-1], prompt_buffer, TERM_WIDTH - 1);
        } else {
            push_line(prompt_buffer);
        }
    }
	
// =============================================================================
// AES-128 ENCRYPTION - GLOBAL (PLACE BEFORE WINDOW CLASS)
// =============================================================================
// AES S-box (256 entries)
static constexpr uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// AES Inverse S-box (256 entries)
static constexpr uint8_t inv_sbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

static constexpr uint8_t rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};


class AES128 {
private:
    uint8_t round_keys[176];
    uint8_t xtime(uint8_t x) { return ((x << 1) ^ (((x >> 7) & 1) * 0x1b)); }
    void key_expansion(const uint8_t* key) {
        memcpy(round_keys, key, 16);
        for (int i = 4; i < 44; i++) {
            uint8_t temp[4];
            memcpy(temp, &round_keys[(i-1)*4], 4);
            if (i % 4 == 0) {
                uint8_t k = temp[0];
                temp[0] = sbox[temp[1]] ^ rcon[i/4];
                temp[1] = sbox[temp[2]];
                temp[2] = sbox[temp[3]];
                temp[3] = sbox[k];
            }
            for (int j = 0; j < 4; j++) round_keys[i*4 + j] = round_keys[(i-4)*4 + j] ^ temp[j];
        }
    }
    void add_round_key(uint8_t* state, int round) {
        for (int i = 0; i < 16; i++) state[i] ^= round_keys[round * 16 + i];
    }
    void sub_bytes(uint8_t* state) { for (int i = 0; i < 16; i++) state[i] = sbox[state[i]]; }
    void inv_sub_bytes(uint8_t* state) { for (int i = 0; i < 16; i++) state[i] = inv_sbox[state[i]]; }
    void shift_rows(uint8_t* state) {
        uint8_t temp;
        temp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
        temp = state[2]; state[2] = state[10]; state[10] = temp;
        temp = state[6]; state[6] = state[14]; state[14] = temp;
        temp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = temp;
    }
    void inv_shift_rows(uint8_t* state) {
        uint8_t temp;
        temp = state[13]; state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = temp;
        temp = state[2]; state[2] = state[10]; state[10] = temp;
        temp = state[6]; state[6] = state[14]; state[14] = temp;
        temp = state[3]; state[3] = state[7]; state[7] = state[11]; state[11] = state[15]; state[15] = temp;
    }
    void mix_columns(uint8_t* state) {
        for (int i = 0; i < 4; i++) {
            uint8_t s0 = state[i*4], s1 = state[i*4+1], s2 = state[i*4+2], s3 = state[i*4+3];
            state[i*4]   = xtime(s0) ^ xtime(s1) ^ s1 ^ s2 ^ s3;
            state[i*4+1] = s0 ^ xtime(s1) ^ xtime(s2) ^ s2 ^ s3;
            state[i*4+2] = s0 ^ s1 ^ xtime(s2) ^ xtime(s3) ^ s3;
            state[i*4+3] = xtime(s0) ^ s0 ^ s1 ^ s2 ^ xtime(s3);
        }
    }
    void inv_mix_columns(uint8_t* state) {
        for (int i = 0; i < 4; i++) {
            uint8_t s0 = state[i*4], s1 = state[i*4+1], s2 = state[i*4+2], s3 = state[i*4+3];
            state[i*4]   = xtime(xtime(xtime(s0) ^ s0) ^ xtime(xtime(s1))) ^ xtime(xtime(s2) ^ s2) ^ xtime(s3) ^ s3;
            state[i*4+1] = xtime(s0) ^ s0 ^ xtime(xtime(xtime(s1) ^ s1) ^ xtime(xtime(s2))) ^ xtime(xtime(s3) ^ s3);
            state[i*4+2] = xtime(xtime(s0) ^ s0) ^ xtime(s1) ^ s1 ^ xtime(xtime(xtime(s2) ^ s2) ^ xtime(xtime(s3)));
            state[i*4+3] = xtime(xtime(xtime(s0))) ^ xtime(xtime(s1) ^ s1) ^ xtime(s2) ^ s2 ^ xtime(xtime(xtime(s3) ^ s3));
        }
    }
public:
    void set_key(const uint8_t* key) { key_expansion(key); }
    void encrypt_block(uint8_t* block) {
        add_round_key(block, 0);
        for (int round = 1; round < 10; round++) {
            sub_bytes(block); shift_rows(block); mix_columns(block); add_round_key(block, round);
        }
        sub_bytes(block); shift_rows(block); add_round_key(block, 10);
    }
    void decrypt_block(uint8_t* block) {
        add_round_key(block, 10);
        for (int round = 9; round > 0; round--) {
            inv_shift_rows(block); inv_sub_bytes(block); add_round_key(block, round); inv_mix_columns(block);
        }
        inv_shift_rows(block); inv_sub_bytes(block); add_round_key(block, 0);
    }
};

void hex_to_bytes(const char* hex, uint8_t* bytes, int len) {
    for (int i = 0; i < len; i++) {
        uint8_t high = hex[i*2], low = hex[i*2+1];
        if (high >= '0' && high <= '9') high = high - '0';
        else if (high >= 'a' && high <= 'f') high = high - 'a' + 10;
        else if (high >= 'A' && high <= 'F') high = high - 'A' + 10;
        if (low >= '0' && low <= '9') low = low - '0';
        else if (low >= 'a' && low <= 'f') low = low - 'a' + 10;
        else if (low >= 'A' && low <= 'F') low = low - 'A' + 10;
        bytes[i] = (high << 4) | low;
    }
}

void bytes_to_hex(const uint8_t* bytes, char* hex, int len) {
    const char* hex_chars = "0123456789abcdef";
    for (int i = 0; i < len; i++) {
        hex[i*2] = hex_chars[(bytes[i] >> 4) & 0xF];
        hex[i*2+1] = hex_chars[bytes[i] & 0xF];
    }
    hex[len*2] = '\0';
}

void pkcs7_pad(uint8_t* data, size_t len) {
    size_t pad_len = 16 - (len % 16);
    for (size_t i = 0; i < pad_len; ++i) data[len + i] = static_cast<uint8_t>(pad_len);
}

bool pkcs7_unpad(uint8_t* data, size_t& len) {
    if (len == 0) return false;
    size_t pad_len = data[len - 1];
    if (pad_len > 16 || pad_len > len) return false;
    for (size_t i = 1; i <= pad_len; ++i) {
        if (data[len - i] != static_cast<uint8_t>(pad_len)) return false;
    }
    len -= pad_len;
    return true;
}

bool aes_encrypt_file(const char* key_hex, const char* infile, const char* outfile) {
    char* content = fat32_read_file_as_string(infile);
    if (!content) return false;
    size_t len = strlen(content);
    size_t padded_len = ((len + 15) / 16) * 16;
    uint8_t* padded = new uint8_t[padded_len];
    memcpy(padded, content, len);
    pkcs7_pad(padded, len);
    uint8_t key[16];
    hex_to_bytes(key_hex, key, 16);
    AES128 aes; aes.set_key(key);
    for (size_t i = 0; i < padded_len / 16; ++i) aes.encrypt_block(padded + i * 16);
    int result = fat32_write_file(outfile, padded, static_cast<uint32_t>(padded_len));
    delete[] padded; delete[] content;
    return result == 0;
}

bool aes_decrypt_file(const char* key_hex, const char* infile, const char* outfile) {
    char* enc_content = fat32_read_file_as_string(infile);
    if (!enc_content) return false;
    size_t enc_len = strlen(enc_content);
    if (enc_len % 16 != 0) { delete[] enc_content; return false; }
    uint8_t* data = reinterpret_cast<uint8_t*>(enc_content);
    uint8_t key[16]; hex_to_bytes(key_hex, key, 16);
    AES128 aes; aes.set_key(key);
    for (size_t i = 0; i < enc_len / 16; ++i) aes.decrypt_block(data + i * 16);
    if (!pkcs7_unpad(data, enc_len)) { delete[] enc_content; return false; }
    int result = fat32_write_file(outfile, data, static_cast<uint32_t>(enc_len));
    delete[] enc_content;
    return result == 0;
}	
static char g_startup_cmd_unused[256];




// Find free process slot
int find_free_elf_slot() {
    for (int i = 0; i < MAX_ELF_PROCESSES; i++) {
        if (!elf_processes[i].active) {
            return i;
        }
    }
    return -1;
}

// Validate ELF header
bool validate_elf_header(const Elf32_Ehdr* ehdr) {
    if (ehdr->e_ident[EI_MAG0] != ELFMAG0 ||
        ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
        ehdr->e_ident[EI_MAG2] != ELFMAG2 ||
        ehdr->e_ident[EI_MAG3] != ELFMAG3) {
        return false;
    }
    
    if (ehdr->e_ident[EI_CLASS] != ELFCLASS32) {
        return false;
    }
    
    if (ehdr->e_ident[EI_DATA] != ELFDATA2LSB) {
        return false;
    }
    
    if (ehdr->e_type != ET_EXEC) {
        return false;
    }
    
    if (ehdr->e_machine != EM_386) {
        return false;
    }
    
    return true;
}


bool disk_has_password() {
    if (!ahci_base || !current_directory_cluster) return false;
    // Probe raw — directory must be readable before unlock.
    bool was_enabled = g_fs_encryption_enabled;
    g_fs_encryption_enabled = false;
    fat_dir_entry_t entry;
    uint32_t sector, offset;
    bool found = (fat32_find_entry(g_disk_password_file, &entry, &sector, &offset) == 0);
    g_fs_encryption_enabled = was_enabled;
    return found;
}
// Kill an ELF process
void kill_elf_process(int slot) {
    if (slot >= 0 && slot < MAX_ELF_PROCESSES && elf_processes[slot].active) {
        ElfProcess& proc = elf_processes[slot];

        // Release the Bochs glue's mapping for this slot BEFORE freeing
        // its backing memory. Without this the glue's SlotState keeps a
        // mem_base pointer into memory we're about to free; the next
        // bochs_activate_slot() on this slot (e.g. a new ELF reusing it)
        // can then dereference that dangling pointer. tick_elf_processes'
        // normal exit path and TerminalWindow::close() both already do
        // this — killelf was the one teardown path that skipped it,
        // which made `killelf` an unsafe way to recover from exactly the
        // runaway/stuck-process situation it exists to handle.
        bochs_release_slot(slot);

        if (proc.memory_base) {
            elf_free_bytes(proc.memory_base);
        }
        if (proc.stack) {
            elf_free_bytes(proc.stack);
        }

        // If a terminal window is still waiting on this slot's input
        // (captured_elf_slot), release it so the prompt comes back
        // instead of the window silently swallowing further keystrokes
        // for a process that no longer exists.
        if (proc.terminal && proc.terminal->get_elf_slot() == slot) {
            proc.terminal->captured_elf_slot = -1;
        }

        proc.active          = false;
        proc.completed        = true;
        proc.cpu_initialized = false;
        proc.memory_base     = nullptr;
        proc.stack           = nullptr;
        proc.memory_size     = 0;
        proc.terminal         = nullptr;
    }
}

// List ELF processes
void list_elf_processes(TerminalWindow* terminal) {
    if (!terminal) return;
    
    terminal->console_print("Active ELF processes:\n");
    bool found = false;
    for (int i = 0; i < MAX_ELF_PROCESSES; i++) {
        if (elf_processes[i].active) {
            char msg[128];
            snprintf(msg, 128, "  Slot %d: entry=0x%x mem=%d KB cmd=%s\n", 
                     i, 
                     elf_processes[i].entry_point,
                     elf_processes[i].memory_size / 1024,
                     elf_processes[i].cmdline);
            terminal->console_print(msg);
            found = true;
        }
    }
    if (!found) {
        terminal->console_print("  (none)\n");
    }
}


// =============================================================================
// DISK PASSWORD SYSTEM
// =============================================================================

static uint32_t simple_hash(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void hash_to_hex(uint32_t hash, char* out) {
    const char* hex = "0123456789abcdef";
    for (int i = 7; i >= 0; i--) {
        out[i] = hex[hash & 0xF];
        hash >>= 4;
    }
    out[8] = '\0';
}
bool disk_check_password(const char* attempt) {
    // Read raw — the password file is always plaintext on disk.
    bool was_enabled = g_fs_encryption_enabled;
    g_fs_encryption_enabled = false;
    char* stored = fat32_read_file_as_string(g_disk_password_file);
    g_fs_encryption_enabled = was_enabled;

    if (!stored) return false;

    uint32_t hash = simple_hash(attempt);
    char hex[9];
    hash_to_hex(hash, hex);
    bool match = (strncmp(stored, hex, 8) == 0);
    delete[] stored;

    if (match) fs_crypto_init(attempt);
    return match;
}
bool disk_set_password(const char* password) {
    if (!password || password[0] == '\0') return false;
    if (strlen(password) < 4) return false;

    // 1. Make sure crypto is OFF so the write goes to disk plaintext.
    //    The password file must always be written unencrypted because
    //    disk_has_password() and disk_check_password() probe it raw.
    bool was_enabled = g_fs_encryption_enabled;
    g_fs_encryption_enabled = false;

    // 2. Compute and write the hash record unencrypted.
    uint32_t hash = simple_hash(password);
    char hex[9];
    hash_to_hex(hash, hex);
    bool ok = (fat32_write_file(g_disk_password_file, hex, 8) == 0);

    // 3. Only arm encryption if the write succeeded.
    if (ok) {
        fs_crypto_init(password);
        g_fs_encryption_enabled = true;
    } else {
        g_fs_encryption_enabled = was_enabled;
        fs_crypto_clear();
    }

    return ok;
}
bool disk_remove_password(const char* current_password) {
    if (!disk_check_password(current_password)) return false;

    // Disarm crypto first so the remove operates on the plaintext directory.
    fs_crypto_clear();
    g_fs_encryption_enabled = false;

    fat32_remove_file(g_disk_password_file);
    g_disk_unlocked = false;
    return true;
}


// Guards all disk-touching commands — returns true if access is allowed
bool disk_access_allowed(TerminalWindow* term) {
    if (!disk_has_password()) return true;   // no password set → open
    if (g_disk_unlocked) return true;         // already authenticated this session
    if (term) term->console_print("Disk is locked. Use: unlock <password>\n");
    return false;
}


// --- Terminal command handler ---
void handle_command() {
    int selected_port = 0;
    char cmd_line[120];
    strncpy(cmd_line, current_line, 119);
    cmd_line[119] = '\0';

    char* command = cmd_line;
    while (*command && *command == ' ') {
        command++;
    }

    if (*command == '\0') {
        if (!in_editor) print_prompt();
        return;
    }

    char* args = command;
    while (*args && *args != ' ') {
        args++;
    }
    if (*args) {
        *args = '\0'; 
        args++;       
        while (*args && *args == ' ') {
            args++;
        }
    }

	if (strcmp(command, "help") == 0) {
		console_print("\nCommands: help, clear, version, time, ps, ls, edit, run, exec,\n"
					  "  compile, rm, cp, mv, formatfs, chkdsk (/r /f), select_disk,\n"
					  "  setpass, removepass, unlock, busybox, pself, killelf,\n"
					  "  killexec, killrun, aesenc, aesdec, test,\n"
					  "  bochs <elf-file> [args]  -- run ELF in Bochs emulator window\n"
				  "  testelf <elf-file>       -- boot ELF via test module (Phase1/2 + diagnostics)\n"
					  "  cc <file.c> [out]        -- compile C in-kernel via TCC\n"
					  "  hello                   -- shortcut: bochs hello\n"
					  "  reset                   -- shortcut: bochs reset\n"
					  "  matrix help             -- NumPy-style arrays + blocked GEMM\n"
					  "  launch <app> | clock | calc | paint | snake | mines\n"
					  "  monitor | inspector | about   -- open desktop apps\n");
	}
	else if (strcmp(command, "bochs") == 0) {
		// Enter Bochs emulator mode in this terminal window. If a
		// filename was given (the documented usage -- see hello_tcc.c's
		// own header comment and the `help` text: "bochs <elf-file>
		// [args]") run it immediately instead of silently discarding
		// the argument and just printing a banner, which is all this
		// branch used to do with anything typed after `bochs`.
		is_emulator_window = true;
		title = "Bochs Emulator";

		char* bochs_fname = nullptr;
		char* bochs_args  = nullptr;
		if (args) {
			char* p = args;
			while (*p == ' ') p++;
			if (*p) {
				// Find the end of the first token BEFORE calling
				// get_arg, which null-terminates the buffer at exactly
				// this boundary -- compute bochs_args first so it
				// already points past that mutation point.
				char* tok_end = p;
				while (*tok_end && *tok_end != ' ') tok_end++;
				char* rest = tok_end;
				while (*rest == ' ') rest++;
				if (*rest) bochs_args = rest;
			}
			bochs_fname = get_arg(args, 0);
		}

		if (!bochs_fname) {
			console_print("=== Bochs i386 CPU emulator ===\n");
			console_print("Just type an ELF filename to run it -- init happens automatically.\n");
		} else {
			console_print("\n");
			int s = load_and_execute_elf(bochs_fname, bochs_args, this);
			if (s >= 0) captured_elf_slot = s;
		}
	}
	else if (strcmp(command, "reset") == 0) {

		bochs_reset_done = true;
		// Run the Bochs CPU reset sequence (test_module_run Phase 1+2)
		// so BX_CPU(0) is fully initialised and the test slab is wiped.
		test_vga_clear();
		g_test_overlay_active = false;
		g_test_overlay_owner  = (void*)this;

		TestSink sink;
		sink.put_line = test_sink_put_line;
		sink.vga_cell = test_sink_vga_cell;
		sink.flush    = test_sink_flush;

		TestResult res;
		res.phase1_ok = 0; res.phase2_ticked = 0;
		res.guest_exit_seen = 0; res.guest_exit_code = 0;
		res.guest_out_len = 0; res.guest_out[0] = 0;

		test_module_run(&sink, &res);

		bochs_reset_all_slots();

		for (int s = 0; s < MAX_ELF_PROCESSES; ++s)
			bochs_register_io_callbacks(s, elf_io_read, elf_io_write, elf_io_exit);

		g_test_overlay_active = false;
		g_test_overlay_owner  = nullptr;

		if (res.phase1_ok && res.guest_exit_seen)
			console_print("reset: OK\n");
		else if (res.phase1_ok)
			console_print("reset: init OK, guest incomplete\n");
		else
			console_print("reset: FAILED\n");
		
	}
	else if (strcmp(command, "testelf") == 0) {
		// Boot a real ELF file from disk through the test module's
		// Phase 1 (BX_CPU init) + Phase 2 (tick) infrastructure, instead
		// of load_and_execute_elf's normal lazy-init path. Useful as a
		// diagnostic: it reuses the same panic-recovery/breadcrumb
		// machinery the `test`/`reset` self-test relies on, with full
		// visibility into init failures, instead of x86_tick's silent
		// per-frame lazy init.
		const char* fname = get_arg(args, 0);
		if (!fname) {
			console_print("Usage: testelf <elf-file>\n");
		} else if ([]{ for (int i = 0; i < MAX_ELF_PROCESSES; ++i)
		                   if (elf_processes[i].active) return true;
		               return false; }()) {
			// testelf shares the same MAX_BOCHS_SLOTS pool as real
			// running ELF processes, hardcodes slot 0 for its own use,
			// and its cleanup calls bochs_reset_all_slots() — which
			// wipes the mapping AND saved CPU snapshot for every slot,
			// not just slot 0. Running it while anything else is live
			// would silently corrupt or kill that process. Refuse
			// instead of doing that quietly.
			console_print("testelf: refusing -- another ELF is currently "
			              "running (use killelf or wait for it to finish)\n");
		} else {
			fat_dir_entry_t entry;
			uint32_t sector = 0, offset = 0;
			if (fat32_find_entry(fname, &entry, &sector, &offset) != 0) {
				console_print("testelf: file not found\n");
			} else {
				char* elfdata = fat32_read_file_as_string(fname);
				if (!elfdata) {
					console_print("testelf: failed to read file\n");
				} else {
					test_vga_clear();
					g_test_overlay_active = false;
					g_test_overlay_owner  = (void*)this;

					TestSink sink;
					sink.put_line = test_sink_put_line;
					sink.vga_cell = test_sink_vga_cell;
					sink.flush    = test_sink_flush;

					TestResult res;
					res.phase1_ok = 0; res.phase2_ticked = 0;
					res.guest_exit_seen = 0; res.guest_exit_code = 0;
					res.guest_out_len = 0; res.guest_out[0] = 0;

					// A real ELF needs far more than the built-in
					// guest's 32-tick budget. 200000 is a soft cap so a
					// genuinely runaway guest can't wedge the terminal
					// forever -- it still blocks the kernel for the
					// duration of the run, same as `reset`/`test`.
					test_module_run_elf(&sink, &res,
						(const unsigned char*)elfdata,
						entry.file_size, 200000);

					delete[] elfdata;

					// Same big-hammer cleanup as `reset`: wipe the test
					// slab's CPU/glue state and re-register the normal
					// per-window ELF I/O callbacks so slot 0 is usable
					// again by the regular load_and_execute_elf path.
					bochs_reset_all_slots();
					for (int s = 0; s < MAX_ELF_PROCESSES; ++s)
						bochs_register_io_callbacks(s, elf_io_read, elf_io_write, elf_io_exit);

					g_test_overlay_active = false;
					g_test_overlay_owner  = nullptr;

					if (!res.phase1_ok)
						console_print("testelf: FAILED (Bochs init)\n");
					else if (res.guest_exit_seen)
						console_print("testelf: guest exited\n");
					else
						console_print("testelf: tick budget exhausted (guest still running)\n");
				}
			}
		}
	}
	else if (strcmp(command, "aesenc") == 0 || strcmp(command, "aesdec") == 0) {
        bool encrypt = strcmp(command, "aesenc") == 0;
        char* key_hex = get_arg(args, 0);
        char* infile = get_arg(args, 1);
        char* outfile = get_arg(args, 2);
        if (!key_hex || !infile || !outfile || strlen(key_hex) != 32) {
            console_print(encrypt ? "Usage: aesenc <32hexkey> <in> <out>\n" : "Usage: aesdec <32hexkey> <in> <out>\n");
            return;
        }
        bool ok = encrypt ? aes_encrypt_file(key_hex, infile, outfile) : aes_decrypt_file(key_hex, infile, outfile);
        console_print(ok ? "AES operation successful.\n" : "AES failed.\n");
    }
	else if (strcmp(command, "select_disk") == 0) {
		g_disk_unlocked = false;
		fs_crypto_clear();                   // wipe key on disk switch
		cmd_list_and_select_disk(args);
		if (disk_has_password())
			console_print("This disk is password protected. Use: unlock <password>\n");
	}

	else if (strcmp(command, "unlock") == 0) {
		if (!disk_has_password()) {
			console_print("No password set. Use: setpass <password>\n");
		} else if (g_disk_unlocked) {
			console_print("Disk already unlocked.\n");
		} else {
			char* pw = get_arg(args, 0);
			if (!pw) {
				console_print("Usage: unlock <password>\n");
			} else if (disk_check_password(pw)) {  // arms crypto internally
				g_disk_unlocked = true;
				console_print("Disk unlocked and decryption armed.\n");
			} else {
				console_print("Wrong password.\n");
			}
		}
	}
	// NOTE: was `if (strcmp(...))` starting a second independent chain;
	// changed to `else if` so commands matched by the first chain (help,
	// aesenc, select_disk, unlock) don't also fall through to the
	// ELF-launch / "command not found" branch at the end.
	else if (strcmp(command, "compile") == 0) {
        cmd_compile(ahci_base, selected_port, get_arg(args, 0));
    }
    // ── cc / tcc — C compiler frontend ───────────────────────────────────
    // cc <source.c> [output]
    //
    // Compilation runs on the HOST via the tcc_glue tool (`make cc`).
    // Inside the kernel this command explains the workflow and, as a
    // convenience, immediately tries to run the ELF if it already exists
    // on the FAT32 disk from a previous host-side `make cc` invocation.
    //
    // Workflow:
    //   1. On the host (before or after booting):
    //        make cc SRC=foo.c          # compiles foo.c → foo on disk.img
    //   2. In this terminal:
    //        cc foo.c                   # (explains the above)
    //        foo                        # runs the compiled ELF via Bochs
    //
    // The reason compilation lives on the host is identical to the reason
    // bochs_glue.so lives there: TCC itself is not a freestanding library
    // and needs malloc, file I/O, and a POSIX environment.
    else if (strcmp(command, "cc")  == 0 ||
             strcmp(command, "tcc") == 0) {
        char* src_arg = get_arg(args, 0);
        char* out_arg = get_arg(args, 1);

        if (!src_arg) {
            console_print("Usage: cc <source.c> [output]\n");
            if (tcc_kernel_version() >= 2) {
                console_print("  Compiles C source from disk to a 32-bit ELF and runs it.\n");
                console_print("  Source must already be on the FAT32 disk (copy via mtools\n");
                console_print("  or write it with the built-in editor if available).\n");
            } else {
                console_print("  In-kernel TCC not linked. Use on the host:\n");
                console_print("      make cc SRC=<file.c>\n");
                console_print("  to compile and inject the ELF into disk.img.\n");
            }
        } else {
            if (tcc_kernel_version() >= 2) {
                // Real in-kernel compilation via libtcc.
                tcc_kernel_cmd_cc(this, src_arg, out_arg);
            } else {
                // Stub path: explain host workflow, auto-run if ELF exists.
                char out_name[64];
                if (out_arg) {
                    strncpy(out_name, out_arg, sizeof(out_name) - 1);
                    out_name[sizeof(out_name) - 1] = '\0';
                } else {
                    strncpy(out_name, src_arg, sizeof(out_name) - 1);
                    out_name[sizeof(out_name) - 1] = '\0';
                    for (int _i = (int)strlen(out_name)-1; _i >= 0; _i--)
                        if (out_name[_i] == '.') { out_name[_i] = '\0'; break; }
                }
                fat_dir_entry_t _elf_entry;
                uint32_t _elf_sec = 0, _elf_off = 0;
                bool _elf_exists =
                    (fat32_find_entry(out_name, &_elf_entry, &_elf_sec, &_elf_off) == 0);

                console_print("cc: in-kernel TCC not available. On the host run:\n");
                console_print("        make cc SRC="); console_print(src_arg);
                if (out_arg) { console_print(" OUT="); console_print(out_arg); }
                console_print("\n");
                
            }
        }
    }
	 else if (strcmp(command, "pself") == 0) {
			// List ELF processes
			list_elf_processes(this);
    } else if (strcmp(command, "killelf") == 0) {
        // Kill an ELF process
        char* arg = get_arg(args, 0);
        if (arg) {
            int slot = simple_atoi(arg);
            kill_elf_process(slot);
            console_print("ELF process killed\n");
        } else {
            console_print("Usage: killelf <slot>\n");
        }
    }
	else if (strcmp(command, "ps") == 0) {
        list_run_processes();
        list_exec_processes();
    }
    else if (strcmp(command, "killrun") == 0) {
        kill_run_process(simple_atoi(get_arg(args, 0)));
    }
    else if (strcmp(command, "killexec") == 0) {
        kill_exec_process(simple_atoi(get_arg(args, 0)));
    }
	
	else if (strcmp(command, "setpass") == 0) {
		// Allowed even when locked so the first password can be set,
		// but changing an existing password requires unlock first.
		if (disk_has_password() && !g_disk_unlocked) {
			console_print("Disk locked. Unlock before changing password.\n");
		} else {
			char* pw = get_arg(args, 0);
			if (!pw || strlen(pw) < 4) {
				console_print("Usage: setpass <password>  (min 4 chars)\n");
			} else {
				if (disk_set_password(pw)) {
					g_disk_unlocked = true; // creator is implicitly unlocked
					console_print("Password set. Disk is now protected.\n");
				} else {
					console_print("Failed to write password file.\n");
				}
			}
		}
	}
	else if (strcmp(command, "removepass") == 0) {
		char* pw = get_arg(args, 0);
		if (!pw) {
			console_print("Usage: removepass <current_password>\n");
		} else if (disk_remove_password(pw)) {
			g_disk_unlocked = false; // reset session state
			console_print("Password removed. Disk is now open.\n");
		} else {
			console_print("Wrong password.\n");
		}
	}
    else if (strcmp(command, "clear") == 0) { line_count = 0; memset(buffer, 0, sizeof(buffer)); }
    else if (strcmp(command, "ls") == 0) { fat32_list_files(); }
    else if (strcmp(command, "edit") == 0) {
        char* filename = get_arg(args, 0);
        if(filename) {
            strncpy(edit_filename, filename, 31);
            edit_filename[31] = '\0';
            in_editor = true;
            edit_current_line = 0;
            edit_cursor_col = 0;
            edit_scroll_offset = 0;
            char* content = fat32_read_file_as_string(filename);
            if (content) {
                int line_count_temp = 1;
                for (char* p = content; *p; p++) if (*p == '\n') line_count_temp++;
                
                edit_lines = new char*[line_count_temp];
                edit_line_count = 0;
                
                char* line_start = content;
                for (char* p = content; *p; p++) {
                    if (*p == '\n') {
                        *p = '\0';
                        edit_lines[edit_line_count] = new char[120];
                        memset(edit_lines[edit_line_count], 0, 120);
                        strncpy(edit_lines[edit_line_count], line_start, 119);
                        edit_line_count++;
                        line_start = p + 1;
                    }
                }
                if (*line_start) {
                    edit_lines[edit_line_count] = new char[120];
                    memset(edit_lines[edit_line_count], 0, 120);
                    strncpy(edit_lines[edit_line_count], line_start, 119);
                    edit_line_count++;
                }
                delete[] content;
            } else {
                edit_lines = new char*[1];
                edit_lines[0] = new char[120];
                memset(edit_lines[0], 0, 120);
                edit_line_count = 1;
            }
        } else {
            console_print("Usage: edit \"<filename>\"\n");
        }
    }
    
    else if (strcmp(command, "rm") == 0) { 
        char* filename = get_arg(args, 0); 
        if(filename) { 
            if(fat32_remove_file(filename) == 0) 
                console_print("File removed.\n"); 
            else 
                console_print("Failed to remove file.\n");
        } else { 
            console_print("Usage: rm \"<filename>\"\n");
        }
    }
    else if (strcmp(command, "cp") == 0) {
        char args_for_src[120];
        strncpy(args_for_src, args, 119);
        char* src = get_arg(args_for_src, 0);

        char args_for_dest[120];
        strncpy(args_for_dest, args, 119);
        char* dest = get_arg(args_for_dest, 1);
        
        if(!src || !dest) { 
            console_print("Usage: cp \"<source>\" \"<dest>\"\n"); 
        } else {
            fat_dir_entry_t entry;
            uint32_t sector, offset;
            if (fat32_find_entry(src, &entry, &sector, &offset) == 0) {
                char* content = new char[entry.file_size];
                if (content && read_data_from_clusters((entry.fst_clus_hi << 16) | entry.fst_clus_lo, content, entry.file_size)) {
                    if(fat32_write_file(dest, content, entry.file_size) == 0) {
                        console_print("Copied.\n");
                    } else {
                        console_print("Write failed.\n");
                    }
                } else {
                    console_print("Read failed.\n");
                }
                if (content) delete[] content;
            } else {
                console_print("Source not found.\n");
            }
        }
    }
    else if (strcmp(command, "mv") == 0) {
        char args_for_src[120];
        strncpy(args_for_src, args, 119);
        char* src = get_arg(args_for_src, 0);

        char args_for_dest[120];
        strncpy(args_for_dest, args, 119);
        char* dest = get_arg(args_for_dest, 1);

        if(!src || !dest) { 
            console_print("Usage: mv \"<source>\" \"<dest>\"\n"); 
        } else {
            if(fat32_rename_file(src, dest) == 0) {
                console_print("Moved.\n");
            } else {
                console_print("Failed. (Source not found or destination exists).\n");
            }
        }
    }
    // CORRECT — format always writes plaintext; encryption is a post-format concern:
	else if (strcmp(command, "formatfs") == 0) {
		bool saved = g_fs_encryption_enabled;
		g_fs_encryption_enabled = false;   // format always writes raw
		fat32_format();
		g_fs_encryption_enabled = saved;
	}
    else if (strcmp(command, "chkdsk") == 0) {
        char* args_copy = new char[120];
        strncpy(args_copy, args, 119);
        args_copy[119] = '\0';
        
        bool fix = false;
        bool fullscan = false;
        
        if (strstr(args_copy, "/f") || strstr(args_copy, "/F")) {
            fix = true;
        }
        if (strstr(args_copy, "/r") || strstr(args_copy, "/R")) {
            fix = true;
            fullscan = true;
        }
        
        chkdsk(fix, true);
        
        if (fullscan) {
            chkdsk_full_scan(fix);
        }
        
        delete[] args_copy;
    }
	
    else if (strcmp(command, "time") == 0) { 
        RTC_Time t = read_rtc(); 
        char buf[64]; 
        snprintf(buf, 64, "%d:%d:%d %d/%d/%d\n", t.hour, t.minute, t.second, t.day, t.month, t.year); 
        console_print(buf); 
    }
    else if (strcmp(command, "version") == 0) { console_print("RTOS++ v1.0 - Robust Parsing\n"); }

    // ── 'bochs <elf>' command ─────────────────────────────────────────────
    // Explicit front-end for running a FAT32 ELF under the Bochs x86
    // emulator.  Always spawns a dedicated emulator TerminalWindow (like
    // the fall-through branch) so the user gets a named window regardless
    // of whether they typed from a shell or an emulator window.
    //
    //   bochs <filename>          — run ELF with no args
    //   bochs <filename> [args…]  — run ELF and pass remaining tokens as args
    //
    // The kernel's Bochs self-test (test_module_run) is NOT re-run here;
    // the CPU is already initialised by kernel_main.  The new emulator
    // window picks up from the lazy-init path in x86_tick exactly the same
    // way as every other ELF launch in this kernel.
    

    // ===================================================================
    // PATCH: matrix array store + desktop suite commands
    // ===================================================================

    // ----- matrix --------------------------------------------------------
    //   matrix create <name> <rows> <cols> [perms=rwx]
    //   matrix list                       (alias: matrix ls)
    //   matrix show <name>
    //   matrix gemm <A> <B> <C>           (C = A . B, blocked tile=4x4)
    //   matrix perms <name> <rwxt>
    //   matrix rm <name>
    else if (strcmp(command, "matrix") == 0) {
        TerminalWindow* term = this;
        char* sub = get_arg(args, 0);
        if (!sub || strcmp(sub, "help") == 0) {
            console_print(
                "matrix: NumPy-style storage arrays + blocked GEMM\n"
                "  matrix create <name> <rows> <cols> [perms=rwx]\n"
                "  matrix list\n"
                "  matrix show   <name>\n"
                "  matrix gemm   <A> <B> <C>      # C = A . B  (i32, tile 4x4)\n"
                "  matrix perms  <name> <rwxt>    # set capability bits\n"
                "  matrix rm     <name>\n"
                "perms: r=read w=write x=kernel-exec t=transmit\n");
            return;
        }
        // append ".npa" if missing
        auto with_ext = [](const char* n, char* out, int cap) -> const char* {
            if (!n || !*n) return nullptr;
            int len = 0; while (n[len] && len < cap - 5) len++;
            bool has_ext = false;
            if (len >= 4 && n[len-4] == '.' &&
                n[len-3] == 'n' && n[len-2] == 'p' && n[len-1] == 'a') has_ext = true;
            for (int i = 0; i < len; ++i) out[i] = n[i];
            if (!has_ext) { out[len++]='.'; out[len++]='n'; out[len++]='p'; out[len++]='a'; }
            out[len] = '\0';
            return out;
        };

        if (strcmp(sub, "create") == 0) {
            char* name = get_arg(args, 1);
            char* rs   = get_arg(args, 2);
            char* cs   = get_arg(args, 3);
            char* ps   = get_arg(args, 4);
            if (!name || !rs || !cs) {
                console_print("Usage: matrix create <name> <rows> <cols> [perms=rwx]\n"); return;
            }
            char nbuf[64]; const char* fn = with_ext(name, nbuf, sizeof(nbuf));
            uint16_t perms = parse_perms(ps);
            int rc = npa_create(fn, (uint32_t)simple_atoi(rs), (uint32_t)simple_atoi(cs), perms, 0);
            if (rc == 0) { console_print("matrix: created "); console_print(fn); console_print("\n"); }
            else { console_print("matrix: create failed (rc=");
                   char b[8]; int_to_string(rc, b); console_print(b); console_print(")\n"); }
            return;
        }
        if (strcmp(sub, "list") == 0 || strcmp(sub, "ls") == 0) {
            const char* names[] = { "A.npa", "B.npa", "C.npa", "D.npa", "E.npa", "F.npa", 0 };
            bool any = false;
            for (int i = 0; names[i]; ++i) {
                ArrayHeader h; void* d = nullptr;
                if (npa_load(names[i], &h, &d) == 0) {
                    npa_print_header(npa_term_print, term, names[i], h);
                    delete[] (uint8_t*)d;
                    any = true;
                }
            }
            if (!any) console_print("matrix: no arrays found (try: matrix create A 8 8)\n");
            return;
        }
        if (strcmp(sub, "show") == 0) {
            char* name = get_arg(args, 1);
            if (!name) { console_print("Usage: matrix show <name>\n"); return; }
            char nbuf[64]; const char* fn = with_ext(name, nbuf, sizeof(nbuf));
            ArrayHeader h; void* d = nullptr;
            int rc = npa_load(fn, &h, &d);
            if (rc != 0) { console_print("matrix: load failed\n"); return; }
            if (!npa_has(h, NPA_R)) {
                console_print("matrix: R denied on "); console_print(fn); console_print("\n");
                delete[] (uint8_t*)d; return;
            }
            npa_print_header(npa_term_print, term, fn, h);
            npa_print_data  (npa_term_print, term, h, d, 8, 8);
            delete[] (uint8_t*)d;
            return;
        }
        if (strcmp(sub, "gemm") == 0) {
            char* a = get_arg(args, 1), *b = get_arg(args, 2), *c = get_arg(args, 3);
            if (!a || !b || !c) { console_print("Usage: matrix gemm <A> <B> <C>\n"); return; }
            char ab[64], bb[64], cb[64];
            const char* fa = with_ext(a, ab, sizeof(ab));
            const char* fb = with_ext(b, bb, sizeof(bb));
            const char* fc = with_ext(c, cb, sizeof(cb));
            int rc = npa_gemm(fa, fb, fc);
            if (rc == 0) { console_print("matrix: gemm OK -> "); console_print(fc); console_print("\n"); }
            else {
                console_print("matrix: gemm failed (rc=");
                char bf[8]; int_to_string(rc, bf); console_print(bf);
                console_print(")  -20=R-denied -21=shape -22=dtype -23=W-denied -24=C-shape\n");
            }
            return;
        }
        if (strcmp(sub, "perms") == 0) {
            char* name = get_arg(args, 1), *ps2 = get_arg(args, 2);
            if (!name || !ps2) { console_print("Usage: matrix perms <name> <rwxt>\n"); return; }
            char nbuf[64]; const char* fn = with_ext(name, nbuf, sizeof(nbuf));
            ArrayHeader h; void* d = nullptr;
            if (npa_load(fn, &h, &d) != 0) { console_print("matrix: load failed\n"); return; }
            h.perms = parse_perms(ps2);
            h.ver_num++;
            int rc = npa_save(fn, &h, d);
            delete[] (uint8_t*)d;
            console_print(rc == 0 ? "matrix: perms updated\n" : "matrix: save failed\n");
            return;
        }
        if (strcmp(sub, "rm") == 0) {
            char* name = get_arg(args, 1);
            if (!name) { console_print("Usage: matrix rm <name>\n"); return; }
            char nbuf[64]; const char* fn = with_ext(name, nbuf, sizeof(nbuf));
            fat32_remove_file(fn);
            console_print("matrix: removed "); console_print(fn); console_print("\n");
            return;
        }
        console_print("matrix: unknown subcommand. Try `matrix help`.\n");
    }
    // ----- desktop suite launchers --------------------------------------
    else if (strcmp(command, "launch") == 0 || strcmp(command, "open") == 0) {
        char* app = get_arg(args, 0);
        if (!app) {
            console_print("Usage: launch <app>\nApps:");
            for (const char** nm = desktop_app_names(); *nm; ++nm) {
                console_print(" "); console_print(*nm);
            }
            console_print("\n");
            return;
        }
        if (!desktop_launch(app, &wm)) {
            console_print("launch: unknown app '"); console_print(app); console_print("'\n");
        }
    }
    else if (strcmp(command, "clock")      == 0) { desktop_launch("clock",   &wm); }
    else if (strcmp(command, "calc")       == 0
          || strcmp(command, "calculator") == 0) { desktop_launch("calc",    &wm); }
    else if (strcmp(command, "paint")      == 0) { desktop_launch("paint",   &wm); }
    else if (strcmp(command, "snake")      == 0) { desktop_launch("snake",   &wm); }
    else if (strcmp(command, "mines")      == 0
          || strcmp(command, "minesweeper")== 0) { desktop_launch("mines",   &wm); }
    else if (strcmp(command, "monitor")    == 0
          || strcmp(command, "sysmon")     == 0
          || strcmp(command, "top")        == 0) { desktop_launch("monitor", &wm); }
    else if (strcmp(command, "inspector")  == 0) { desktop_launch("matrix",  &wm); }
    else if (strcmp(command, "about")      == 0) { desktop_launch("about",   &wm); }

    // Fall-through: try to handle 'command' as an ELF file from FAT32.
    // The ELF runs inside the Bochs CPU emulator via x86_tick / cpu_loop.
    //
    // This used to require the user to type `bochs` (enter emulator
    // mode) and then `reset` (force-run the Bochs init/self-test
    // sequence) before an ELF filename would actually do anything.
    // kernel_run_global_ctors_once() now guarantees BX_CPU(0)/bx_mem
    // are constructed before kernel_main ever reaches init_elf_system(),
    // and init_elf_system() already registers IO callbacks for every
    // slot at boot — so no per-window init is actually required for
    // correctness here. (bochs_reset_done used to also gate a call to
    // bochs_reset_all_slots() here, but that call is GLOBAL — it wipes
    // every slot's mapping and CPU snapshot, not just this window's —
    // so doing it on a per-window "first ELF run" basis meant any
    // window's first launch could silently kill every other window's
    // already-running ELF process. Removed; nothing here actually
    // needed it.)
    else {

        is_emulator_window = true;
        title = "Bochs Emulator";

        // Run the ELF in-place; output flows to this window's
        // console_print via the elf_io_write callback chain.
	console_print("\n");
        int s = load_and_execute_elf(command, args, this);
        if (s >= 0) captured_elf_slot = s;
    }

    if(!in_editor) print_prompt();
}
int load_and_execute_elf(const char* filename, const char* args, TerminalWindow* terminal) {
    char* elfdata = fat32_read_file_as_string(filename);
    if (!elfdata) {
        if (terminal) terminal->console_print("Failed to read ELF file\n");
        return -1;
    }

    int result = -1;
    uint8_t* mem = nullptr;
    uint8_t* stack = nullptr;
    ElfProcess* proc = nullptr;
    int slot = -1;

    do {
        fat_dir_entry_t entry;
        uint32_t sector = 0, offset = 0;
        if (fat32_find_entry(filename, &entry, &sector, &offset) != 0) {
            if (terminal) terminal->console_print("ELF: directory entry not found\n");
            break;
        }

        Elf32_Ehdr ehdr;
        memcpy(&ehdr, elfdata, sizeof(Elf32_Ehdr));
        if (!validate_elf_header(&ehdr)) {
            if (terminal) terminal->console_print("Invalid ELF file\n");
            break;
        }

        Elf32_Phdr* phdr = (Elf32_Phdr*)(elfdata + ehdr.e_phoff);
        uint32_t filesize = entry.file_size;

        slot = find_free_elf_slot();
        if (slot < 0) {
            if (terminal) terminal->console_print("No free ELF slot\n");
            break;
        }

        proc = &elf_processes[slot];
        *proc = ElfProcess();
        proc->terminal = terminal;
        // ── Scrub per-slot I/O ring buffers ──────────────────────────────
        // `*proc = ElfProcess()` value-initialises POD members, but the
        // 512-byte inbuf and 4096-byte outbuf char arrays are NOT in the
        // struct's default initialiser list and therefore retain whatever
        // bytes the previous run left in them when this slot is reused.
        // The ring-buffer HEAD/TAIL indices get reset to 0 by the line
        // above (they have default initialisers), so the stale bytes are
        // not directly "readable" via pop_input/pop_output — but they
        // remain reachable through any code path that reads inbuf[]
        // beyond the wrap (e.g. snprintf-style buffer dumps, or a guest
        // doing speculative IN-port reads). They are also a forensic
        // trip-hazard when the head/tail get out of sync due to an
        // unrelated bug — the "phantom" bytes look like real input.
        //
        // Empirically the bug manifested as the third+ in-place run of
        // `hello` in the same emulator window producing
        // "HELLO WOHELLO WO..." loops with the user's typed "hello"
        // prefix concatenated to the guest output. The kernel-side
        // teardown in tick_elf_processes was freeing memory_base/stack
        // and calling bochs_reset_all_slots(), but never wiping the
        // input/output ring buffers — so when the same slot was reused,
        // it carried forward bytes from the previous run's interaction.
        // Zero-fill makes slot reuse architecturally identical to
        // first-time use.
        for (int _b = 0; _b < INBUFSIZE;  ++_b) proc->inbuf[_b]  = 0;
        for (int _b = 0; _b < OUTBUFSIZE; ++_b) proc->outbuf[_b] = 0;
        proc->in_head = proc->in_tail = proc->out_head = proc->out_tail = 0;
        if (args) {
            strncpy(proc->cmdline, args, sizeof(proc->cmdline) - 1);
            proc->cmdline[sizeof(proc->cmdline) - 1] = 0;
        }

        bool found_load = false;
        uint32_t minvaddr = 0xFFFFFFFFu;
        uint32_t maxvaddr = 0;
        bool phdr_bad = false;

        for (int i = 0; i < ehdr.e_phnum; i++) {
            if (phdr[i].p_type != PT_LOAD) continue;
            // Skip zero-memsz PT_LOAD entries. Some linkers (incl. TCC's own
            // linker) emit placeholder/alignment PT_LOAD headers with
            // p_vaddr == 0 and p_memsz == 0. Letting these count toward
            // minvaddr silently drags minvaddr down to 0 even though the
            // *real* code/data segments load at a sane address (e.g.
            // 0x08002000). That in turn makes `minvaddr - ELF_SLOT_RESERVE`
            // underflow to 0xFFFFF000, which is exactly the bogus
            // vaddr_base disk_guest_ptr() then rejects every mailbox
            // address against. Mirrors the same skip already done in
            // load_elf_image_to_slab().
            if (phdr[i].p_memsz == 0) continue;
            found_load = true;
            if (phdr[i].p_memsz < phdr[i].p_filesz) { phdr_bad = true; break; }
            if (phdr[i].p_offset + phdr[i].p_filesz > filesize) { phdr_bad = true; break; }
            if (phdr[i].p_vaddr < minvaddr) minvaddr = phdr[i].p_vaddr;
            uint32_t end = phdr[i].p_vaddr + phdr[i].p_memsz;
            if (end > maxvaddr) maxvaddr = end;
        }

        if (phdr_bad) {
            if (terminal) terminal->console_print("ELF: malformed program header (memsz/filesz/offset)\n");
            break;
        }
        if (!found_load) {
            if (terminal) terminal->console_print("ELF: no PT_LOAD segment found\n");
            break;
        }
        if (maxvaddr <= minvaddr) {
            if (terminal) terminal->console_print("ELF: empty or inverted load range\n");
            break;
        }

        uint32_t imgsize = maxvaddr - minvaddr;
        if (imgsize == 0 || imgsize > 6 * 1024 * 1024) {
            if (terminal) terminal->console_print("ELF: image size out of range (0 or >6MB)\n");
            break;
        }

        // Reserve a dead-zone buffer offset before the real image so
        // bochs_set_process_memory's table injection (GDT/IDT/stub,
        // written into the first ~0x900 bytes of whatever buffer it's
        // given) can NEVER land on the program's own code -- regardless
        // of whether the ELF was linked with one of the project's
        // special "slab_reserved" linker scripts (hello.ld/guest.ld/
        // tcc_guest.ld) or with a normal toolchain that knows nothing
        // about this kernel's memory layout. Without this, any ELF
        // whose lowest PT_LOAD segment lands at buffer offset 0 -- which
        // is every ELF, unless its own linker script manually pads --
        // has its first instructions silently corrupted before it ever
        // runs. vaddr_base shifts down by the same RESERVE amount, so
        // the guest-visible absolute address of the real image (and
        // therefore every address the compiler/linker baked into the
        // program) is completely unchanged -- only the underlying
        // buffer layout moves.
        const uint32_t ELF_SLOT_RESERVE = 0x1000;

        uint32_t memsize = imgsize + ELFHEAPSIZE + ELF_SLOT_RESERVE;
        if (memsize > 6 * 1024 * 1024) {
            if (terminal) terminal->console_print("ELF: image+heap exceeds 6MB limit\n");
            break;
        }


        // minvaddr - ELF_SLOT_RESERVE underflows (wraps to ~0xFFFFF000) if
        // the lowest real PT_LOAD segment sits below the 4 KiB reserve —
        // that bogus vaddr_base then makes every legitimate guest address
        // (e.g. mailbox/buffer pointers, which are small numbers relative
        // to the program's actual load address) look "out of range" to
        // disk_guest_ptr(), silently dropping every disk command. Catch it
        // here instead of wrapping into a huge, wrong base.
        if (minvaddr < ELF_SLOT_RESERVE) {
            if (terminal) terminal->console_print("ELF: lowest PT_LOAD vaddr is too low to reserve a slab header (linker script/-Ttext issue?)\n");
            break;
        }

        mem = elf_alloc_bytes(memsize);
        if (!mem) {
            if (terminal) terminal->console_print("ELF: out of memory (image)\n");
            break;
        }
        memset(mem, 0, memsize);
        proc->memory_base = mem;
        proc->memory_size = memsize;
        proc->vaddr_base = minvaddr - ELF_SLOT_RESERVE;
        proc->vaddr_end = maxvaddr;

        bool copy_bad = false;
        for (int i = 0; i < ehdr.e_phnum; i++) {
            if (phdr[i].p_type != PT_LOAD) continue;
            uint32_t dstoff = ELF_SLOT_RESERVE + (phdr[i].p_vaddr - minvaddr);
            if (dstoff + phdr[i].p_filesz > memsize) { copy_bad = true; break; }
            memcpy(mem + dstoff, elfdata + phdr[i].p_offset, phdr[i].p_filesz);
            if (phdr[i].p_memsz > phdr[i].p_filesz) {
                memset(mem + dstoff + phdr[i].p_filesz, 0, phdr[i].p_memsz - phdr[i].p_filesz);
            }
        }
        if (copy_bad) {
            if (terminal) terminal->console_print("ELF: segment offset out of bounds\n");
            break;
        }

        stack = elf_alloc_bytes(ELFSTACKSIZE);
        if (!stack) {
            if (terminal) terminal->console_print("ELF: out of memory (stack)\n");
            break;
        }
        memset(stack, 0, ELFSTACKSIZE);
        proc->stack = stack;

        proc->entry_point = ehdr.e_entry;
        proc->eip = proc->entry_point;
        // ESP sits at the TOP of the slab, growing down. The previous
        // formula was `vaddr_base + memory_size - ELFHEAPSIZE - 16`,
        // which evaluated to `vaddr_base + imgsize - 16` — i.e. just
        // BELOW the end of the loaded image, INSIDE .rodata/.data.
        // For a small hello-world that mostly worked because only a
        // handful of pushes happened before the program exited, but
        // it was timing/luck dependent: the first push of %ebp at
        // [esp-4] could clobber whatever was at slab offset
        // imgsize-20, and a longer or differently-laid-out program
        // could see the program's own data corrupted by the stack.
        //
        // Correct layout (slab grows up; addresses increase →):
        //     [ image (text/rodata/data/bss) ][ heap/stack arena ]
        //     ^vaddr_base                    ^brk                ^ESP
        // The brk grows up from end-of-image; ESP grows down from the
        // top. They share the ELFHEAPSIZE-byte arena and collide only
        // if the program both heap-allocates a lot AND nests deep.
        // The dedicated 64 KB `stack` allocation (proc->stack) is
        // historical scratch — it is NOT wired to ESP.
        proc->esp = proc->vaddr_base + proc->memory_size - 16;
        proc->active = true;
        proc->cpu_initialized = false;
        proc->waiting_for_input = false;
        proc->completed = false;
        proc->exit_code = 0;

        result = slot;
    } while (0);

    if (result < 0) {
        if (proc) {
            if (proc->stack) { elf_free_bytes(proc->stack); proc->stack = nullptr; }
            if (proc->memory_base) { elf_free_bytes(proc->memory_base); proc->memory_base = nullptr; }
            proc->active = false;
            proc->cpu_initialized = false;
            proc->waiting_for_input = false;
            proc->completed = false;
            proc->exit_code = 0;
            proc->memory_size = 0;
            proc->entry_point = 0;
            proc->vaddr_base = 0;
            proc->vaddr_end = 0;
            proc->esp = 0;
            proc->eip = 0;
            proc->cmdline[0] = 0;
        }
    }

    delete[] elfdata;
    return result;
}


public:
    // Public wrapper so tcc_kernel.cpp's bridge can launch an ELF without
    // needing access to the private load_and_execute_elf directly.
    int exec_elf(const char* filename, const char* args) {
        return load_and_execute_elf(filename, args, this);
    }

    // is_emulator_window=true marks this terminal as a window that was
    // spawned specifically to host a Bochs-emulated ELF. Used by
    // handle_command()'s fall-through branch to decide whether to run an
    // unknown ELF *in-place* (we are the emulator window) or to *spawn a
    // fresh emulator window* (we are an ordinary shell).
    TerminalWindow(int x, int y, const char* startup_command = nullptr,
                   bool emulator_mode = false)
        : Window(x, y, 640, 400, emulator_mode ? "Bochs Emulator" : "Terminal"),
          line_count(0), line_pos(0), in_editor(false),
          edit_lines(nullptr), edit_line_count(0), edit_current_line(0),
          edit_cursor_col(0), edit_scroll_offset(0),
          prompt_visual_lines(0),
          is_emulator_window(emulator_mode) {
        memset(buffer, 0, sizeof(buffer));
        current_line[0] = '\0';
        private_startup_cmd[0] = '\0';
        if (startup_command) {
            strncpy(private_startup_cmd, startup_command, 255);
            private_startup_cmd[255] = '\0';
        }

        // Print a banner in the emulator window so it's obvious to the
        // user that this is the Bochs CPU emulator booting their ELF, not
        // a normal shell. The banner is queued onto the terminal buffer
        // before the auto-startup command runs on the next update().
        if (emulator_mode) {
            console_print("=== Bochs i386 CPU emulator ===\n");
            console_print("Loading ELF and entering protected mode...\n");
        }

        update_prompt_display(); // Show the initial prompt
    }
    bool is_emulator_window = false;
    bool bochs_reset_done   = false;  // reset runs once per window
    int captured_elf_slot = -1;
    int get_elf_slot() const override { return captured_elf_slot; }

    void close() override {
        // If this window owned the Bochs self-test overlay, relinquish it
        // so g_test_overlay_owner never dangles after we are deleted.
        if (g_test_overlay_owner == (void*)this) {
            g_test_overlay_owner  = nullptr;
            g_test_overlay_active = false;
        }
        // Kill the attached ELF process so it disappears from the taskbar.
        // Also release the Bochs glue slot and free the slab/stack to avoid
        // dangling pointers in the memory-handler table and memory leaks.
        if (captured_elf_slot >= 0 && captured_elf_slot < MAX_ELF_PROCESSES) {
            ElfProcess& proc = elf_processes[captured_elf_slot];
            proc.active    = false;
            proc.completed = true;
            proc.terminal  = nullptr;
            // Unregister the Bochs memory handlers for this slot before
            // freeing the slab — otherwise the glue holds a dangling
            // mem_base pointer that any future activate_slot could deref.
            bochs_release_slot(captured_elf_slot);
            if (proc.memory_base) { elf_free_bytes(proc.memory_base); proc.memory_base = nullptr; }
            if (proc.stack)       { elf_free_bytes(proc.stack);       proc.stack       = nullptr; }
            proc.memory_size     = 0;
            proc.cpu_initialized = false;
        }
        captured_elf_slot = -1;
        is_closed = true;
    }

    ~TerminalWindow() { 
        if(edit_lines) {
            for(int i = 0; i < edit_line_count; i++) delete[] edit_lines[i];
            delete[] edit_lines;
        }
    }

    // Map a VGA attribute byte (bg<<4 | fg) to a framebuffer RGB color.
    // Only the foreground nibble drives the glyph color; we use a compact
    // 16-entry CGA-style palette. The background nibble tints the row.
    static uint32_t test_vga_palette(uint8_t nibble) {
        static const uint32_t pal[16] = {
            0x000000, 0x0000AA, 0x00AA00, 0x00AAAA,
            0xAA0000, 0xAA00AA, 0xAA5500, 0xAAAAAA,
            0x555555, 0x5555FF, 0x55FF55, 0x55FFFF,
            0xFF5555, 0xFF55FF, 0xFFFF55, 0xFFFFFF
        };
        return pal[nibble & 0x0F];
    }

    // Draw the three g_test_vga[] rows at the top of the content area.
    // Returns the vertical pixel offset the terminal text should shift by
    // so it sits below the overlay.
    int render_test_overlay() {
        const int cell_w  = 8;             // font glyph width
        const int row_h   = 10;            // overlay row height
        const int top     = y + 28;        // just under the titlebar
        const int max_col = (w - 10) / cell_w < 80 ? (w - 10) / cell_w : 80;

        for (int r = 0; r < 3; ++r) {
            int row_y = top + r * row_h;
            // Tint strip: use the background nibble of the row's first
            // non-blank cell (cells in a row share a background).
            uint8_t bg_nib = 0;
            for (int c = 0; c < max_col; ++c) {
                if (g_test_vga[r][c].ch != ' ') {
                    bg_nib = (g_test_vga[r][c].attr >> 4) & 0x0F;
                    break;
                }
            }
            draw_rect_filled(x + 4, row_y, max_col * cell_w, row_h,
                             test_vga_palette(bg_nib));
            for (int c = 0; c < max_col; ++c) {
                char ch = g_test_vga[r][c].ch;
                if (ch == ' ' || ch == 0) continue;
                uint32_t fg = test_vga_palette(g_test_vga[r][c].attr & 0x0F);
                char s[2] = { ch, 0 };
                draw_string(s, x + 5 + c * cell_w, row_y + 1, fg);
            }
        }
        return 3 * row_h + 2;              // shift terminal text down
    }

    void draw() override {
        if (!has_focus && is_closed) return;

        using namespace ColorPalette;
        
        uint32_t titlebar_color = has_focus ? TITLEBAR_ACTIVE : TITLEBAR_INACTIVE;
        draw_rect_filled(x, y, w, 25, titlebar_color);
        // Show slot info in titlebar when running an ELF
        if (captured_elf_slot >= 0) {
            char ttl[48];
            ttl[0]='T'; ttl[1]='e'; ttl[2]='r'; ttl[3]='m'; ttl[4]=' ';
            ttl[5]='['; ttl[6]='S'; ttl[7]='0'+(char)captured_elf_slot; ttl[8]=':'; ttl[9]=' ';
            const char* cmd = elf_processes[captured_elf_slot].cmdline;
            int ci = 10;
            for (int k = 0; cmd[k] && ci < 42; k++, ci++) ttl[ci] = cmd[k];
            ttl[ci++] = ']'; ttl[ci] = 0;
            draw_string(ttl, x + 5, y + 8, TEXT_WHITE);
        } else {
            draw_string(title, x + 5, y + 8, TEXT_WHITE);
        }

        draw_rect_filled(x + w - 22, y + 4, 18, 18, BUTTON_CLOSE);
        draw_string("X", x + w - 17, y + 8, TEXT_WHITE);

        draw_rect_filled(x, y + 25, w, h - 25, WINDOW_BG);

        for (int i = 0; i < w; i++) put_pixel_back(x + i, y, WINDOW_BORDER);
        for (int i = 0; i < w; i++) put_pixel_back(x + i, y + h - 1, WINDOW_BORDER);
        for (int i = 0; i < h; i++) put_pixel_back(x, y + i, WINDOW_BORDER);
        for (int i = 0; i < h; i++) put_pixel_back(x + w - 1, y + i, WINDOW_BORDER);

        // ── Bochs self-test VGA overlay ─────────────────────────────────
        // When this terminal activated the `test` module, paint the three
        // VGA-style rows (breadcrumbs / fault tag / GUEST line) the module
        // wrote into g_test_vga[]. This reproduces test_main.cpp's 0xB8000
        // overlay inside the window. The terminal text is shifted down by
        // overlay_dy so it doesn't collide with the three rows.
        int overlay_dy = 0;
        if (!in_editor && g_test_overlay_active &&
            g_test_overlay_owner == (void*)this) {
            overlay_dy = render_test_overlay();
        }

        if (!in_editor) {
    // How many 10px text lines actually fit between the content-area
    // top (y + 30 + overlay_dy) and the window's bottom border (y + h).
    // The old code hard-coded `i < 38`, which — combined with the test
    // overlay shifting text down by overlay_dy — drew lines past the
    // bottom border ("overhanging by one"). Compute the real capacity.
    int content_top = 30 + overlay_dy;
    int visible_rows = (h - content_top) / 10;
    if (visible_rows < 1) visible_rows = 1;

    // When the buffer holds more lines than fit, show the most recent
    // ones (tail) rather than the oldest (head) — otherwise fresh guest
    // output scrolls off the bottom and stale text stays pinned at top.
    int first = line_count - visible_rows;
    if (first < 0) first = 0;

    for (int i = first; i < line_count; i++) {
        int screen_row = i - first;
        draw_string(buffer[i], x + 5, y + content_top + screen_row * 10,
                    ColorPalette::TEXT_WHITE);
    }
} else {
    for (int row = 0; row < EDIT_ROWS; ++row) {
        int line_idx = edit_scroll_offset + row;
        int y_line = y + 30 + row * EDIT_LINE_PIX;

        if (line_idx < edit_line_count) {
            if (line_idx == edit_current_line) {
                draw_rect_filled(x + 2, y_line, w - 4, EDIT_LINE_PIX, ColorPalette::TEXT_GRAY);
            }
            draw_string(edit_lines[line_idx], x + 5, y_line, ColorPalette::TEXT_WHITE);
        }
    }

    if ((g_timer_ticks / 15) % 2 == 0 && edit_current_line >= edit_scroll_offset &&
        edit_current_line < edit_scroll_offset + EDIT_ROWS) {
        int visible_row = edit_current_line - edit_scroll_offset;
        int cursor_x = x + 5 + edit_cursor_col * EDIT_COL_PIX;
        int cursor_y = y + 30 + visible_row * EDIT_LINE_PIX;
        draw_rect_filled(cursor_x, cursor_y, EDIT_COL_PIX, EDIT_LINE_PIX, ColorPalette::CURSOR_WHITE);
    }
}
    }

    void on_key_press(char c) override {
    if (in_editor) {
        if (!edit_lines || edit_current_line >= edit_line_count) return;

        char* current_line_ptr = edit_lines[edit_current_line];
        size_t current_len = strlen(current_line_ptr);

        if (c == 17 || c == 27) { // Ctrl+Q or ESC to save and exit
            int total_len = 0;
            for (int i = 0; i < edit_line_count; i++) {
                total_len += strlen(edit_lines[i]) + 1;
            }
            char* file_content = new char[total_len + 1];
            if (!file_content) return;
            file_content[0] = '\0';
            for (int i = 0; i < edit_line_count; i++) {
                strcat(file_content, edit_lines[i]);
                if (i < edit_line_count - 1) {
                   strcat(file_content, "\n");
                }
            }
            fat32_write_file(edit_filename, file_content, strlen(file_content));
            delete[] file_content;
            in_editor = false;
            console_print("File saved.\n");
            return;
        } 
        else if (c == KEY_UP) {
            if (edit_current_line > 0) edit_current_line--;
        } 
        else if (c == KEY_DOWN) {
            if (edit_current_line < edit_line_count - 1) edit_current_line++;
        } 
        else if (c == KEY_LEFT) {
            if (edit_cursor_col > 0) edit_cursor_col--;
        } 
        else if (c == KEY_RIGHT) {
            if (edit_cursor_col < (int)current_len) edit_cursor_col++;
        } 
        else if (c == KEY_HOME) {
            edit_cursor_col = 0;
        }
        else if (c == KEY_END) {
            edit_cursor_col = current_len;
        }
        else if (c == KEY_DELETE) {
            if (edit_cursor_col < (int)current_len) {
                memmove(&current_line_ptr[edit_cursor_col], 
                       &current_line_ptr[edit_cursor_col + 1], 
                       current_len - edit_cursor_col);
            } else if (edit_current_line < edit_line_count - 1) {
                // Delete at end of line - join with next line
                char* next_line_ptr = edit_lines[edit_current_line + 1];
                if (current_len + strlen(next_line_ptr) < TERM_WIDTH - 1) {
                    strcat(current_line_ptr, next_line_ptr);
                    editor_delete_line_at(edit_current_line + 1);
                }
            }
        }
        else if (c == '\n') { // Enter key
            const char* right_part_text = &current_line_ptr[edit_cursor_col];
            editor_insert_line_at(edit_current_line + 1, right_part_text);
            current_line_ptr[edit_cursor_col] = '\0';
            edit_current_line++;
            edit_cursor_col = 0;
        } 
        else if (c == '\b') { // Backspace
            if (edit_cursor_col > 0) {
                memmove(&current_line_ptr[edit_cursor_col - 1], 
                       &current_line_ptr[edit_cursor_col], 
                       current_len - edit_cursor_col + 1);
                edit_cursor_col--;
            } else if (edit_current_line > 0) {
                int prev_line_idx = edit_current_line - 1;
                char* prev_line_ptr = edit_lines[prev_line_idx];
                int prev_len = strlen(prev_line_ptr);
                if (prev_len + current_len < TERM_WIDTH - 1) {
                    strcat(prev_line_ptr, current_line_ptr);
                    editor_delete_line_at(edit_current_line);
                    edit_current_line = prev_line_idx;
                    edit_cursor_col = prev_len;
                }
            }
        } 
        else if (c >= 32 && c < 127) { // Printable characters
            // **WORD WRAP IMPLEMENTATION**
            const int MAX_LINE_WIDTH = 75; // Characters before wrap
            
            if (current_len < TERM_WIDTH - 2) {
                // Insert character
                memmove(&current_line_ptr[edit_cursor_col + 1], 
                       &current_line_ptr[edit_cursor_col], 
                       current_len - edit_cursor_col + 1);
                current_line_ptr[edit_cursor_col] = c;
                edit_cursor_col++;
                
                // Check if line is too long and needs wrapping
                int new_len = strlen(current_line_ptr);
                if (new_len > MAX_LINE_WIDTH) {
                    // Find last space to wrap at
                    int wrap_pos = MAX_LINE_WIDTH;
                    bool found_space = false;
                    
                    for (int i = MAX_LINE_WIDTH; i > MAX_LINE_WIDTH - 20 && i > 0; i--) {
                        if (current_line_ptr[i] == ' ') {
                            wrap_pos = i;
                            found_space = true;
                            break;
                        }
                    }
                    
                    // If no space found near margin, force wrap at max width
                    if (!found_space) {
                        wrap_pos = MAX_LINE_WIDTH;
                    }
                    
                    // Create wrapped text for next line
                    char wrapped_text[TERM_WIDTH];
                    memset(wrapped_text, 0, TERM_WIDTH);
                    strcpy(wrapped_text, &current_line_ptr[wrap_pos]);
                    
                    // Trim leading space from wrapped text
                    char* trimmed = wrapped_text;
                    while (*trimmed == ' ') trimmed++;
                    
                    // Truncate current line at wrap point
                    current_line_ptr[wrap_pos] = '\0';
                    
                    // Insert wrapped text as new line
                    editor_insert_line_at(edit_current_line + 1, trimmed);
                    
                    // Move cursor to next line if it was past wrap point
                    if (edit_cursor_col > wrap_pos) {
                        edit_current_line++;
                        edit_cursor_col = edit_cursor_col - wrap_pos;
                        // Account for trimmed spaces
                        while (edit_cursor_col > 0 && wrapped_text[0] == ' ') {
                            edit_cursor_col--;
                        }
                        if (edit_cursor_col < 0) edit_cursor_col = 0;
                    }
                }
            }
        }
        
        editor_clamp_cursor_to_line();
        editor_ensure_cursor_visible();
        return; // END OF EDITOR HANDLING
    }	else {
		
		    // BUSYBOX CAPTURE
			if (captured_elf_slot >= 0) {
			// Echo + feed
			// Only echo printable bytes + newline/tab. Echoing arbitrary
			// key codes (modifier-key scancodes, arrow keys, etc.) drops
			// bytes 0-31 / 127 into the terminal buffer where they render
			// as blank spaces in font.h — the same mechanism that produced
			// the "HELL O" rendering bug. The raw byte still goes to the
			// ELF's stdin via push_input below, so applications that want
			// to interpret special keys still see them.
			unsigned char uc = (unsigned char)c;
			if (uc == '\n' || uc == '\t' || (uc >= 32 && uc < 127)) {
				char echo[2] = {c, 0};
				console_print(echo);
			}
			push_input(captured_elf_slot, c);
			if (c == '\n') elf_processes[captured_elf_slot].waiting_for_input = false;
			return;
		}
            if (c == '\n') {
                // run_contexts[] is indexed by its own slot, NOT by window index.
                // There is no 1:1 mapping between windows and run slots, so we
                // just always treat Enter as a command submission.
                prompt_visual_lines = 0;
                handle_command();
                line_pos = 0;
                current_line[0] = '\0';
                update_prompt_display();
            }
			
			else if (c == '\b') {
                if (line_pos > 0) {
                    line_pos--;
                    current_line[line_pos] = 0;
                }
                update_prompt_display();
            } else if (c >= 32 && c < 127 && line_pos < TERM_WIDTH - 2) {
                current_line[line_pos++] = c;
                current_line[line_pos] = 0;
                update_prompt_display();
            }
        }
    }

     // --- THIS IS THE CORRECTED UPDATE METHOD ---
    void update() override {
        // Check if there is a startup command waiting to be executed
        if (private_startup_cmd[0] != '\0') {
            strncpy(current_line, private_startup_cmd, TERM_WIDTH - 1);
            current_line[TERM_WIDTH - 1] = '\0';
            private_startup_cmd[0] = '\0';
            push_line(current_line);
            handle_command();
            line_pos = 0;
            current_line[0] = '\0';
            update_prompt_display();
        }
    }


    void console_print(const char* s) override {
        if (!s || in_editor) return;

        // ── Sanitize input ────────────────────────────────────────────────
        // Filter to printable ASCII plus the whitespace we explicitly handle
        // (\n, \t). Without this, any byte 0-31 or 127 in `s` (uninitialized
        // stack memory, echoed modifier keys, stray emulator garbage, etc.)
        // gets stuffed verbatim into the terminal buffer. Those bytes have
        // empty glyphs in font.h, so they RENDER AS BLANK SPACES — visually
        // identical to U+0020. This is the root cause of bugs like
        // "HELLO" displaying as "HELL O": a stray control byte landed in
        // the buffer between L and O. \r is converted to \n for safety;
        // \b is intentionally NOT supported here (the legitimate guest
        // output paths don't emit it, and accepting it would let stray
        // 0x08 bytes erase real output).
        char clean[8192];
        int  cn = 0;
        for (const char* p = s; *p && cn < (int)sizeof(clean) - 1; ++p) {
            unsigned char c = (unsigned char)*p;
            if (c == '\n' || c == '\t')             clean[cn++] = (char)c;
            else if (c == '\r')                     clean[cn++] = '\n';
            else if (c >= 32 && c < 127)            clean[cn++] = (char)c;
            // else: drop (would render blank and corrupt the display).
        }
        clean[cn] = '\0';
        if (cn == 0) return;

        int saved_prompt_lines = prompt_visual_lines;
        if (saved_prompt_lines > 0) {
            remove_last_n_lines(saved_prompt_lines);
            prompt_visual_lines = 0;
        }

        push_wrapped_text(clean, term_cols_cont());
        update_prompt_display();
    }
};

// NpaPrint adapter — forwards into a TerminalWindow*. Forward-declared
// above so the matrix command handler can take its address before
// TerminalWindow is complete; defined here now that the type is whole.
void npa_term_print(void* ctx, const char* s) {
    static_cast<TerminalWindow*>(ctx)->console_print(s);
}

// TestSink::put_line implementation. Defined here because it needs the
// complete TerminalWindow type to route text into the window that owns
// the test overlay. Falls back silently if no terminal owns it.
extern "C" void test_sink_put_line(const char* s) {
    if (!s) return;
    if (g_test_overlay_owner) {
        ((TerminalWindow*)g_test_overlay_owner)->console_print(s);
    }
}

void WindowManager::execute_context_menu_action(int item_index) {
    if (item_index < 0 || item_index >= num_context_menu_items) return;
    const char* action = context_menu_items[item_index];

    if (current_context == CTX_DESKTOP) {
        if (strcmp(action, "File Explorer") == 0) {
            launch_new_explorer();
        } else if (strcmp(action, "Paste") == 0) {
            if (g_clipboard_buffer[0] != '\0') {
                const char* src_path = g_clipboard_buffer;
                const char* filename = strrchr(src_path, '/');
                filename = filename ? filename + 1 : src_path;
                
                char new_name[32] = "copy_of_";
                strncat(new_name, filename, 22);

                fat32_copy_file(src_path, new_name);
                load_desktop_items();
            }
        }
    }
    else if (current_context == CTX_ICON) {
        DesktopItem& item = desktop_items[context_icon_idx];
        
        if (strcmp(action, "Run") == 0) {
            char command_buffer[128];
            snprintf(command_buffer, 128, "run %s", item.name);
            launch_terminal_with_command(command_buffer);
        } else if (strcmp(action, "Edit") == 0) {
            char command_buffer[128];
            snprintf(command_buffer, 128, "edit \"%s\"", item.name);
            launch_terminal_with_command(command_buffer);
        } else if (strcmp(action, "Copy") == 0) {
            strncpy(g_clipboard_buffer, item.path, 1023);
        } else if (strcmp(action, "Delete") == 0) {
            fat32_remove_file(item.path);
            load_desktop_items();
        }
    }
    else if (current_context == CTX_EXPLORER_ITEM) {
        const char* filename = context_file_path;

        if (strcmp(action, "Run") == 0) {
            char command_buffer[128];
            snprintf(command_buffer, 128, "run %s", filename);
            launch_terminal_with_command(command_buffer);
        } else if (strcmp(action, "Edit") == 0) {
            char command_buffer[128];
            snprintf(command_buffer, 128, "edit \"%s\"", filename);
            launch_terminal_with_command(command_buffer);
        } else if (strcmp(action, "Create Shortcut") == 0) {
            char shortcut_name[32];
            char shortcut_content[128];
            
            strncpy(shortcut_name, filename, 27);
            char* dot = strrchr(shortcut_name, '.');
            if (dot) *dot = '\0';
            strcat(shortcut_name, ".lnk");

            snprintf(shortcut_content, 128, "run %s", filename);

            fat32_write_file(shortcut_name, shortcut_content, strlen(shortcut_content));
            load_desktop_items();
        } 
    }

    context_menu_active = false;
}

void WindowManager::handle_input(char key, int mx, int my, bool left_down, bool left_clicked, bool right_clicked) {
    // --- Static variables to track double-clicks ---
    static uint32_t last_click_tick = 0;
    static int last_click_icon_idx = -1;
    const uint32_t DOUBLE_CLICK_SPEED = 20; // Ticks to wait for a double click

    // --- 1. Handle Context Menu Clicks ---
    if (context_menu_active && left_clicked) {
        int menu_width = 150;
        int item_height = 20;
        if (mx > context_menu_x && mx < context_menu_x + menu_width) {
            int item_index = (my - context_menu_y) / item_height;
            if (item_index >= 0 && item_index < num_context_menu_items) {
                execute_context_menu_action(item_index);
                return; // Action taken, end input handling
            }
        }
        context_menu_active = false; // Clicked outside, close menu
    }

    if (context_menu_active && right_clicked) {
        context_menu_active = false;
        return;
    }

    // --- 2. Handle Dragging ---
    if (dragging_idx != -1) { // Dragging a window
        if (left_down) {
            windows[dragging_idx]->x = mx - drag_offset_x;
            windows[dragging_idx]->y = my - drag_offset_y;
        } else {
            dragging_idx = -1;
        }
        return;
    }
    if (dragging_icon_idx != -1) { // Dragging an icon
        if (left_down) {
            desktop_items[dragging_icon_idx].x = mx - drag_offset_x;
            desktop_items[dragging_icon_idx].y = my - drag_offset_y;
        } else {
            dragging_icon_idx = -1;
        }
        return;
    }
    
    // --- 3. Handle Right Clicks (Opening Context Menu) ---
    if (right_clicked) {
		if (focused_idx != -1) {
            Window* win = windows[focused_idx];
            if (mx >= win->x && mx < win->x + win->w && my >= win->y && my < win->y + win->h) {
                win->on_mouse_right_click(mx, my);
                return; // The window handled the click
            }
        }
        // First, check if a click happened on a desktop icon
        int clicked_icon_index = -1;
        for (int i = num_desktop_items - 1; i >= 0; --i) {
            if (mx >= desktop_items[i].x && mx < desktop_items[i].x + 40 &&
                my >= desktop_items[i].y && my < desktop_items[i].y + 50) {
                clicked_icon_index = i; // Save the index of the clicked icon
                break; // Found it, no need to check others
            }
        }

        if (clicked_icon_index != -1) {
            // A desktop icon was right-clicked
            context_menu_active = true;
            context_menu_x = mx;
            context_menu_y = my;
            current_context = CTX_ICON;
            context_icon_idx = clicked_icon_index; // Use the saved index
            num_context_menu_items = 0;

            // Check if it's an executable
            if (strstr(desktop_items[clicked_icon_index].name, ".obj") != nullptr || strstr(desktop_items[clicked_icon_index].name, ".OBJ") != nullptr) {
                context_menu_items[num_context_menu_items++] = "Run";
            }
            
            context_menu_items[num_context_menu_items++] = "Edit"; // ADDED THIS LINE
            context_menu_items[num_context_menu_items++] = "Copy";
            context_menu_items[num_context_menu_items++] = "Delete";

        } else {
            // No icon was clicked, this is a right-click on the desktop itself
            context_menu_active = true;
            context_menu_x = mx;
            context_menu_y = my;
            current_context = CTX_DESKTOP;
            num_context_menu_items = 0;
            context_menu_items[num_context_menu_items++] = "File Explorer";
            context_menu_items[num_context_menu_items++] = "Paste";
        }
        return;
    }

    // --- 4. Handle Left Clicks (Dragging, Opening, Focusing) ---
    if (left_clicked) {
        // Check window interactions first (top to bottom)
        for (int i = num_windows - 1; i >= 0; i--) {
            if (mx >= windows[i]->x && mx < windows[i]->x + windows[i]->w &&
                my >= windows[i]->y && my < windows[i]->y + windows[i]->h) {
                
                set_focus(i);
                if (windows[i]->is_in_close_button(mx, my)) {
                    windows[i]->close();
                } else if (windows[i]->is_in_titlebar(mx, my)) {
                    dragging_idx = focused_idx;
                    drag_offset_x = mx - windows[dragging_idx]->x;
                    drag_offset_y = my - windows[dragging_idx]->y;
                } else {
                    windows[i]->on_mouse_click(mx, my);
                }
                return;
            }
        }

        // Check icon interactions (double-click and drag start)
        for (int i = num_desktop_items - 1; i >= 0; --i) {
            if (mx >= desktop_items[i].x && mx < desktop_items[i].x + 32 &&
                my >= desktop_items[i].y && my < desktop_items[i].y + 45) {

                // Check for a double-click
                if (last_click_icon_idx == i && (g_timer_ticks - last_click_tick) < DOUBLE_CLICK_SPEED) {
                    // Double-click detected!
                    DesktopItem& item = desktop_items[i]; // Use a reference for cleaner code

					if (strcmp(item.path, "explorer.app") == 0) {
						launch_new_explorer();
					} 
					// This part handles executing .obj files
					else if (strstr(item.name, ".obj") != nullptr || strstr(item.name, ".OBJ") != nullptr) {
						char command_buffer[128];
						snprintf(command_buffer, 128, "run %s", item.name);
						launch_terminal_with_command(command_buffer);
					}
                    
                    // Reset double-click tracking
                    last_click_tick = 0;
                    last_click_icon_idx = -1;
                } else {
                    // This is a first click, start dragging and set up for double-click
                    dragging_icon_idx = i;
                    drag_offset_x = mx - desktop_items[i].x;
                    drag_offset_y = my - desktop_items[i].y;
                    last_click_icon_idx = i;
                    last_click_tick = g_timer_ticks;
                }
                return;
            }
        }

        // Check taskbar button clicks — layout mirrors draw_desktop()
        if (my >= (int)fb_info.height - 36 && my <= (int)fb_info.height - 4) {
            int btn_w = 80;
            int bx = 4;
            // "Terminal" launcher
            if (mx >= bx && mx < bx + btn_w) {
                launch_new_terminal();
                return;
            }
            bx += btn_w + 4;
            // Per-slot ELF buttons
            for (int s = 0; s < MAX_ELF_PROCESSES; ++s) {
                if (!elf_processes[s].active) continue;  // skip idle and completed slots
                if (mx >= bx && mx < bx + btn_w) {
                    // Find the terminal window that owns this slot and bring it to front
                    bool found = false;
                    for (int wi = 0; wi < num_windows; ++wi) {
                        if (windows[wi]->get_elf_slot() == s) {
                            set_focus(wi);
                            found = true;
                            break;
                        }
                    }
                    // Terminal was closed but ELF still active — open a new one attached to slot
                    if (!found && elf_processes[s].active) {
                        static int _tw_ctr = 0;
                        int off = (_tw_ctr++ % 10) * 30;
                        TerminalWindow* tw2 = new TerminalWindow(100 + off, 50 + off);
                        tw2->captured_elf_slot = s;
                        elf_processes[s].terminal = tw2;
                        wm.add_window(tw2);
                    }
                    return;
                }
                bx += btn_w + 4;
                if (bx + btn_w >= (int)fb_info.width - 100) break;
            }
        }

        // If nothing was clicked, reset double-click tracking
        last_click_icon_idx = -1;
    }

    // --- 5. Handle Keyboard Input ---
    if (key != 0 && focused_idx != -1 && focused_idx < num_windows)
        windows[focused_idx]->on_key_press(key);
}

void WindowManager::print_to_focused(const char* s) {
    if (focused_idx != -1 && focused_idx < num_windows) 
        windows[focused_idx]->console_print(s);
}

void launch_new_terminal() {
    static int win_count = 0;
    int idx = (win_count++ % 10);
    int off = idx * 30;
    wm.add_window(new TerminalWindow(100 + off, 50 + off));
}


void launch_new_explorer() {
    static int win_count = 0;
    int idx = (win_count++ % 10);
    int off = idx * 30;
    wm.add_window(new FileExplorerWindow(120 + off, 70 + off, "/"));
}


// ADD THIS NEW FUNCTION
void launch_terminal_with_command(const char* command) {
    static int win_count = 0;
    int idx = (win_count++ % 10);
    int off = idx * 30;
    wm.add_window(new TerminalWindow(150 + off, 90 + off, command));
}

void swap_buffers() {
    // Safety: never blit to VGA text region or below 16MB physical
    if (!fb_info.ptr || !backbuffer) return;
    if ((uintptr_t)fb_info.ptr < 0x1000000u) return;
    uint32_t pitch_pixels = fb_info.pitch / 4;  // pitch is in bytes, convert to pixels
    if (pitch_pixels == fb_info.width) {
        // Fast path: pitch matches width, single blit
        uint32_t* dest = fb_info.ptr;
        uint32_t* src = backbuffer;
        size_t count = fb_info.width * fb_info.height;
        asm volatile (
            "rep movsl"
            : "=S"(src), "=D"(dest), "=c"(count)
            : "S"(src), "D"(dest), "c"(count)
            : "memory"
        );
    } else {
        // Pitch != width: copy row by row respecting stride
        for (uint32_t y = 0; y < fb_info.height; y++) {
            uint32_t* dest = (uint32_t*)((uint8_t*)fb_info.ptr + y * fb_info.pitch);
            uint32_t* src  = backbuffer + y * fb_info.width;
            uint32_t  count = fb_info.width;
            asm volatile (
                "rep movsl"
                : "=S"(src), "=D"(dest), "=c"(count)
                : "S"(src), "D"(dest), "c"(count)
                : "memory"
            );
        }
    }
}

// TestSink::flush implementation. test_module_run() blocks the kernel
// main loop for the entire duration of a test, so the normal per-frame
// wm.update_all() + swap_buffers() pass never gets to run. The module
// calls this between breadcrumbs and around blocking calls so the
// overlay is painted and pushed to the screen live — and so a hang
// inside the Bochs glue still leaves a frame on screen showing exactly
// how far the test got. It mirrors the main loop's paint sequence.
extern "C" void test_sink_flush(void) {
    g_gfx.clear_screen(ColorPalette::DESKTOP_GRAY );
    wm.update_all();                       // draws windows incl. overlay
    draw_cursor(mouse_x, mouse_y, ColorPalette::CURSOR_WHITE);
    draw_vga_overlay();                    // framebuffer breadcrumb rows
    swap_buffers();                        // push frame to the display
}

static volatile bool g_evt_timer = false;
static volatile bool g_evt_input = false;
static volatile bool g_evt_dirty = true;
// This is now defined before TerminalWindow to resolve the dependency
// static volatile uint32_t g_timer_ticks = 0;

extern "C" void idle_signal_timer() { g_evt_timer = true; g_timer_ticks++; }
extern "C" void idle_signal_input() { g_evt_input = true; }
extern "C" void mark_screen_dirty() { g_evt_dirty = true; }

static void init_screen_timer(uint16_t hz) {
    uint16_t divisor = 1193182 / hz;
    outb(0x43, 0x36);
    outb(0x40, divisor & 0xFF);
    outb(0x40, (divisor >> 8) & 0xFF);
}

// =============================================================================
// KERNEL MAIN - ATOMIC FRAME RENDERING
// =============================================================================



struct BochsCPURegs {
    uint32_t eax, ebx, ecx, edx, esi, edi, esp, ebp;
};

// Forward decls for I/O callback adapters defined just below.
static int  elf_io_read (int slot);
static void elf_io_write(int slot, char c);
static void elf_io_exit (int slot, int code);

// Replace init_elf_system() call in kernel_main with:
void init_elf_system() {
    for (auto& p : elf_processes) {
        p.active          = false;
        p.cpu_initialized = false;
        p.in_head  = p.in_tail  = 0;
        p.out_head = p.out_tail = 0;
    }
    // Register IO callbacks for every slot. These are what make
    // port-0xE9 writes from the guest reach the terminal: the chain is
    //   guest: out 0xE9, al
    //   -> bx_devices_c::outp (bochs_infra.cpp)
    //   -> bochs_guest_putc   (bochs_glue.cpp)
    //   -> write_cb = elf_io_write
    //   -> push_output(slot, c)
    //   -> tick_elf_processes drains it to terminal->console_print
    for (int s = 0; s < MAX_ELF_PROCESSES; ++s) {
        bochs_register_io_callbacks(s, elf_io_read, elf_io_write, elf_io_exit);
    }
    // NOTE: bochs_cpu_init() is NOT called here.
    //
    // We tried calling bochs_cpu_prewarm() (a guarded BX_CPU::initialize)
    // here to surface init-time bugs early, but it triggered the VMware
    // BAR1-unmap path documented at the top of vmware_svga_init(), causing
    // VMware to die with "execute an invalid part of memory". The early
    // call was unnecessary anyway: the host IDT installed in boot.S means
    // a host fault inside BX_CPU::initialize() during lazy init now
    // produces a visible "!XX" breadcrumb at VGA row 1 instead of a
    // silent triple fault.
    //
    // Init runs lazily on the first ELF launch via x86_tick. The
    // bochs_cpu_init() guard added in bochs_glue.cpp ensures
    // BX_CPU::initialize() runs at most once per boot regardless of how
    // many times the kernel calls bochs_cpu_init().
}
// I/O callback adapters
static int  elf_io_read (int slot) { return pop_input(slot); }
static void elf_io_write(int slot, char c) { push_output(slot, c); }
static void elf_io_exit (int slot, int code) {
    if (slot >= 0 && slot < MAX_ELF_PROCESSES) {
        // DIAGNOSTIC: report the exit code (now the real fault vector
        // number, per the per-vector stub change in inject_slab_tables).
        // The buffer is zero-initialized this time, unlike the old
        // disabled version this comment used to warn about.
        char buf[64] = {0};
        snprintf(buf, sizeof(buf), "\n[guest exit, code=%d]\n", code);
        console_print(buf);

        elf_processes[slot].completed = true;
        elf_processes[slot].active    = false;
    }
}static void dbg(const char* s) { /* write to VGA or terminal */ }
// Minimal ELF execution path for Bochs-backed processes.
// Assumes your existing kernel includes/types/helpers are already present.


static inline unsigned int align_down(unsigned int v, unsigned int a) {
    return v & ~(a - 1);
}

static inline unsigned int align_up(unsigned int v, unsigned int a) {
    return (v + a - 1) & ~(a - 1);
}

static bool load_elf_image_to_slab(
    int slot,
    const unsigned char* elf,
    unsigned int elf_size,
    unsigned int& entry_out)
{
    if (elf_size < 52) return false;
    if (!(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F')) return false;
    if (elf[4] != 1 || elf[5] != 1) return false;

    auto rd16 = [&](unsigned int off) -> unsigned short {
        return (unsigned short)(elf[off] | (elf[off + 1] << 8));
    };
    auto rd32 = [&](unsigned int off) -> unsigned int {
        return (unsigned int)(elf[off] |
                              (elf[off + 1] << 8) |
                              (elf[off + 2] << 16) |
                              (elf[off + 3] << 24));
    };

    unsigned short phoff = rd32(28);
    unsigned short phentsize = rd16(42);
    unsigned short phnum = rd16(44);
    unsigned int entry = rd32(24);

    if (phoff == 0 || phentsize < 32 || phnum == 0) return false;

    unsigned int min_vaddr = 0xFFFFFFFFu;
    unsigned int max_vaddr = 0;

    for (unsigned int i = 0; i < phnum; ++i) {
        unsigned int p = phoff + i * phentsize;
        if (p + 32 > elf_size) return false;

        unsigned int p_type = rd32(p + 0);
        if (p_type != 1) continue; // PT_LOAD

        unsigned int p_vaddr = rd32(p + 8);
        unsigned int p_memsz = rd32(p + 20);

        if (p_memsz == 0) continue;

        if (p_vaddr < min_vaddr) min_vaddr = p_vaddr;
        if (p_vaddr + p_memsz > max_vaddr) max_vaddr = p_vaddr + p_memsz;
    }

    if (min_vaddr == 0xFFFFFFFFu || max_vaddr <= min_vaddr) return false;

    unsigned int vaddr_base = align_down(min_vaddr, 0x1000);
    unsigned int vaddr_top = align_up(max_vaddr + ELF_STACK_SIZE + ELF_HEAP_SIZE, 0x1000);
    unsigned int slab_size = vaddr_top - vaddr_base;

    unsigned char* slab = elf_alloc_bytes(slab_size);
    if (!slab) return false;
    memset(slab, 0, slab_size);

    for (unsigned int i = 0; i < phnum; ++i) {
        unsigned int p = phoff + i * phentsize;
        if (p + 32 > elf_size) continue;

        unsigned int p_type = rd32(p + 0);
        if (p_type != 1) continue;

        unsigned int p_offset = rd32(p + 4);
        unsigned int p_vaddr  = rd32(p + 8);
        unsigned int p_filesz  = rd32(p + 16);
        unsigned int p_memsz   = rd32(p + 20);

        if (p_offset + p_filesz > elf_size) {
            elf_free_bytes(slab);
            return false;
        }

        unsigned int dest = p_vaddr - vaddr_base;
        if (dest + p_memsz > slab_size) {
            elf_free_bytes(slab);
            return false;
        }

        memcpy(slab + dest, elf + p_offset, p_filesz);
        if (p_memsz > p_filesz) {
            memset(slab + dest + p_filesz, 0, p_memsz - p_filesz);
        }
    }

    entry_out = entry;
    bochs_activate_slot(slot);
    bochs_set_process_memory(slab, slab_size, vaddr_base);
    bochs_finalize_process_memory();

    // Populate ElfProcess so that start_elf_process (and kill_elf_process)
    // can track and free this slab correctly.
    ElfProcess& proc = elf_processes[slot];
    proc.memory_base = slab;
    proc.memory_size = slab_size;
    proc.vaddr_base  = vaddr_base;
    proc.vaddr_end   = vaddr_top;

    return true;
}

static bool start_elf_process(int slot, const unsigned char* elf, unsigned int elf_size) {
    ElfProcess& proc = elf_processes[slot];
    unsigned int entry = 0;

    // load_elf_image_to_slab calls bochs_activate_slot + bochs_set_process_memory
    // and populates proc.memory_base / proc.memory_size / proc.vaddr_base.
    if (!load_elf_image_to_slab(slot, elf, elf_size, entry)) return false;

    proc.active = true;
    proc.completed = false;
    proc.entry_point = entry;

    // vaddr_base was set by load_elf_image_to_slab; don't clobber it with a
    // page-aligned guess derived from the entry point (those can differ for
    // PIE or non-zero-based binaries).
    unsigned int stack_top = proc.vaddr_base + proc.memory_size;
    proc.esp = stack_top - 16;
    proc.brk_addr = proc.vaddr_base + proc.memory_size - ELF_HEAP_SIZE;

    // Finish CPU wiring: init once, then point at entry.
    bochs_cpu_init();
    bochs_cpu_set_esp(proc.esp);
    bochs_cpu_set_eip(proc.entry_point);
    bochs_set_brk(slot, proc.brk_addr);

    // Mark cpu_initialized so x86_tick skips the lazy-init path and goes
    // straight to bochs_cpu_tick (we've done the full setup just above).
    proc.cpu_initialized = true;

    return true;
}

extern "C" volatile unsigned char bx_panic_breadcrumbs[64];
// ── Diagnostic breadcrumb at VGA row 2 ────────────────────────────────────
// Each x86_tick lazy-init step writes a tag char to row 2 columns 0..N.
// Tags are:
//   col 0: 'L'  — entered lazy init
//   col 1: 'M'  — set_process_memory returned
//   col 2: 'I'  — bochs_cpu_init returned
//   col 3: 'S'  — set_esp returned
//   col 4: 'E'  — set_eip returned
//   col 5: 'B'  — set_brk returned
//   col 6: 'T'  — about to call bochs_cpu_tick
//   col 7: 't'  — bochs_cpu_tick returned
// If lazy init wedges, the last tag visible tells you which call did it.
//
// We write to BOTH the VGA text-mode plane (forensic for pmemsave) AND
// directly to the live framebuffer (visible to the user even if the
// kernel never reaches swap_buffers again). Direct framebuffer writes
// bypass the back buffer and use fb_info.ptr; that means they survive
// a hang inside x86_tick that prevents the next paint cycle.
static inline void x86_breadcrumb(int col, char c) {
    if (col < 0 || col >= 80) return;

    // 1. VGA text-mode plane (forensic).
    {
        volatile unsigned short* p =
            (volatile unsigned short*)(0xB8000 + 2 * 80 + 2 * col);
        *p = (unsigned short)(0x0E00u | (unsigned char)c);
    }

    // 2. Live framebuffer (visible). 8x8 yellow glyph at row 2 of overlay.
    if (!fb_info.ptr) return;
    if ((unsigned char)c > 127) return;
    const uint8_t* glyph = font + (int)c * 8;
    int x0 = col * 8;
    int y0 = 16;          // row 2 of overlay (rows 0,1,2 each 8px tall)
    if (x0 + 8 > (int)fb_info.width)  return;
    if (y0 + 8 > (int)fb_info.height) return;

    uint32_t color = 0xFFFF55u;   // bright yellow
    uint32_t bg    = 0x000000u;
    for (int yy = 0; yy < 8; ++yy) {
        uint32_t* row = &fb_info.ptr[(y0 + yy) * (fb_info.pitch / 4) + x0];
        uint8_t bits = glyph[yy];
        for (int xx = 0; xx < 8; ++xx) {
            row[xx] = (bits & (0x80 >> xx)) ? color : bg;
        }
    }
}

static bool x86_tick(int slot, int steps) {
    ElfProcess& proc = elf_processes[slot];
    // Tick a process that is alive: active and NOT yet completed.
    // The previous form (`!proc.active || !proc.completed`) was inverted —
    // it returned for every live process, so the guest never advanced and
    // its port-0xE9 output never reached the terminal display.
    if (!proc.active || proc.completed) return false;

    if (!proc.cpu_initialized) {
        if (!proc.memory_base || proc.memory_size == 0) {
            proc.active    = false;
            proc.completed = true;
            return false;
        }
        // Correct init order:
        //   1. bochs_cpu_init()          -- global Bochs one-time init (idempotent)
        //   2. bochs_activate_slot()     -- select which slot g_active_slot points at
        //   3. bochs_set_process_memory  -- resets CPU to clean state, injects GDT/IDT/
        //                                   stub tables, then restores to protected mode
        //   4. set_esp / set_eip         -- point the protected-mode CPU at guest entry
        //
        // The previous order was: activate → set_memory → cpu_init.
        // bochs_set_process_memory does BX_CPU::reset() + slot_restore_cpu() to
        // arrive at protected mode. bochs_cpu_init() called *after* does another
        // BX_CPU::reset() — wiping that protected mode state back to real mode.
        // The guest then runs (or tries to) in real mode at a 32-bit EIP, which
        // either executes garbage or hangs inside cpu_loop with no visible output.
        x86_breadcrumb(0, 'L');
        bochs_cpu_init();
        x86_breadcrumb(1, 'I');
        bochs_activate_slot(slot);
        bochs_set_process_memory(proc.memory_base, proc.memory_size,
                                 proc.vaddr_base);
        x86_breadcrumb(2, 'M');
        bochs_cpu_set_esp(proc.esp);
        x86_breadcrumb(3, 'S');
        bochs_cpu_set_eip(proc.entry_point);
        x86_breadcrumb(4, 'E');
        bochs_set_brk(slot, proc.vaddr_base + proc.memory_size - ELFHEAPSIZE);
        x86_breadcrumb(5, 'B');
        proc.cpu_initialized = true;
    } else {
        bochs_activate_slot(slot);
    }

    x86_breadcrumb(6, 'T');

    // Diagnostic: snapshot EIP BEFORE tick so we can detect an unexpected
    // jump back to entry after the tick. (Don't enable unconditionally —
    // gated on the very-recent-restart condition below so normal output
    // isn't polluted.)
    uint32_t eip_before = bochs_cpu_get_eip();

bochs_cpu_tick(steps);
x86_breadcrumb(7, 't');

bool just_started_waiting =
    bochs_process_wants_input(slot) && in_empty(slot);

if (just_started_waiting) {
    proc.waiting_for_input = true;
    // The guest yielded mid-`IN` on port 0xE7. EIP isn't reliably
    // retired across that abort, so don't trust it this tick —
    // skip the exit/EIP-range checks below entirely and pick back
    // up cleanly next tick once real input has been pushed.
    return true;
}

// ─── FIX: detect clean guest-exit signal ──────────────────────────────
if (!proc.active || proc.completed) return false;

unsigned int eip = bochs_cpu_get_eip();
if (eip == 0 || eip < proc.vaddr_base || eip >= proc.vaddr_base + proc.memory_size) {
    proc.completed = true;
    proc.active    = false;
    return false;
}

return true;

    return true;
}
// And definition without default:
void tick_elf_processes(int steps) {
    bool any_exited_this_frame = false;

    for (int i = 0; i < MAX_ELF_PROCESSES; ++i) {
        ElfProcess& proc = elf_processes[i];
        // Drain output and step a process that is alive: active and not
        // completed. Inverting this gate (the old `!active || !completed`)
        // skipped every live process, leaving the terminal display silent
        // even though the Bochs guest had written bytes to port 0xE9.
        if (!proc.active || proc.completed) continue;

        while (!out_empty(i)) {
            char tmp[256];
            int n = 0;
            while (!out_empty(i) && n < 255) tmp[n++] = pop_output(i);
            tmp[n] = 0;
            if (proc.terminal && n) {
                proc.terminal->console_print(tmp);
                // Guest produced output. Flag the frame dirty so the main
                // loop actually repaints. Without this the repaint is
                // gated on g_evt_dirty / hasNewInput — both only set by
                // user input — so guest output sat invisibly in the
                // terminal buffer until the next keypress ("need to press
                // enter to print the last lot").
                g_evt_dirty = true;
            }
        }

        if (proc.waiting_for_input && in_empty(i)) continue;

        bool running = x86_tick(i, steps);

        while (!out_empty(i)) {
            char tmp[256];
            int n = 0;
            while (!out_empty(i) && n < 255) tmp[n++] = pop_output(i);
            tmp[n] = 0;
            if (proc.terminal && n) {
                proc.terminal->console_print(tmp);
                g_evt_dirty = true;   // see note above
            }
        }

        if (!running) {
            proc.active = false;
            proc.completed = true;
            if (proc.terminal) proc.terminal->captured_elf_slot = -1;
            // A process just finished — the prompt needs to come back and
            // any final output needs to show. Repaint this frame.
            g_evt_dirty = true;

            // ── Per-slot kernel-side teardown ────────────────────────
            // Free the slab and stack we allocated in load_and_execute_elf,
            // and clear cpu_initialized so the NEXT process that lands in
            // this slot re-runs the lazy-init path from scratch.
            //
            // We do NOT touch the Bochs glue here. The glue side gets a
            // single heavy `bochs_reset_all_slots()` call at the end of
            // this function (see below) — that wipes every per-slot
            // mapping AND hardware-resets BX_CPU(0), so launch N starts
            // from the same state as launch 1. Trying to keep the glue
            // "incrementally consistent" with surgical per-slot updates
            // had a long history of subtle bugs (the second `bochs hello`
            // looping "HELLO WOHELLO WO..." because some untracked CPU
            // field survived across slot reuse). The big-hammer reset
            // is cheap and unambiguous.
            //
            // EXCEPTION (added for concurrent processes): if any OTHER
            // slot is still active, bochs_reset_all_slots() below will
            // be SKIPPED (resetting the live CPU would clobber that
            // peer's mid-execution state). In that case we still need
            // to clear THIS slot's mapping and dangling mem_base pointer
            // before freeing the slab — otherwise the glue carries
            // forward a dangling pointer that could be dereferenced on
            // a future activate_slot. bochs_release_slot does that
            // surgically without touching peer slots or BX_CPU(0).
            // It's safe to call unconditionally: when the all-slots
            // reset path also fires below, release_slot's wipe is
            // simply overwritten by the same values via reset_all_slots.
            bochs_release_slot(i);
            if (proc.memory_base) { elf_free_bytes(proc.memory_base); proc.memory_base = nullptr; }
            if (proc.stack)       { elf_free_bytes(proc.stack);       proc.stack       = nullptr; }
            proc.memory_size     = 0;
            proc.cpu_initialized = false;

            // ── Wipe per-slot I/O ring buffers and transient flags ──
            // Belt-and-braces companion to the scrub in
            // load_and_execute_elf: cleared HERE on the exit edge so
            // the slot is left in a strictly-empty state the instant
            // the process becomes inactive. If any code path on the
            // next frame reads from inbuf/outbuf before the next
            // load_and_execute_elf runs (e.g. a stray
            // tick_elf_processes pass on a slot whose `completed`
            // flag flipped but whose `active` flag hasn't been
            // re-set yet), it sees a clean empty queue rather than
            // stale bytes from the run that just ended.
            //
            // Without this, the third in-place run of `hello` in the
            // same emulator window produced "HELLO WOHELLO WO..."
            // loops with leftover keystrokes from previous runs
            // bleeding into the new guest's stdin and the previous
            // guest's tail-end output bleeding into the new
            // terminal display.
            for (int _b = 0; _b < INBUFSIZE;  ++_b) proc.inbuf[_b]  = 0;
            for (int _b = 0; _b < OUTBUFSIZE; ++_b) proc.outbuf[_b] = 0;
            proc.in_head           = 0;
            proc.in_tail           = 0;
            proc.out_head          = 0;
            proc.out_tail          = 0;
            proc.waiting_for_input = false;
            proc.exit_code         = 0;
            proc.input_pos         = 0;

            any_exited_this_frame = true;
        }
    }

    // ── Deferred glue-wide reset ────────────────────────────────────
    // If any process exited this frame AND no other Bochs-emulated
    // process is still running, wipe Bochs's glue state back to its
    // post-boot baseline. The next launch will then start from the
    // same state as launch #1.
    //
    // CRITICAL: do NOT reset while another slot is still running.
    // bochs_reset_all_slots() unmaps every slab and hardware-resets
    // BX_CPU(0); doing that to a live slot drops its EIP/CRs/segments
    // back to the post-reset state. On the very next frame, x86_tick
    // would re-enter the lazy-init path and bochs_cpu_set_eip would
    // restart the guest from _start. The visible symptom is a runaway
    // window spamming "HELLO WOHELLO WO..." for as long as another
    // window is finishing — exactly because every frame's exit on
    // window B triggered a reset that restarted window A from the top.
    if (any_exited_this_frame) {
        // Only wipe the Bochs glue when ALL slots are done AND all
        // output has been drained. Use two separate loops: the old
        // single-loop version broke out early on the first active slot,
        // so any_output_pending was never checked for remaining slots.
        bool any_still_active   = false;
        bool any_output_pending = false;
        for (int j = 0; j < MAX_ELF_PROCESSES; ++j) {
            if (elf_processes[j].active) { any_still_active = true; break; }
        }
        if (!any_still_active) {
            for (int j = 0; j < MAX_ELF_PROCESSES; ++j) {
                if (!out_empty(j)) { any_output_pending = true; break; }
            }
        }
        if (!any_still_active && !any_output_pending) {
            bochs_reset_all_slots();
        }
    }
     
}

extern "C" void cmd_exec(const char* code_text) {
    if (!code_text) return;
    TCompiler C;
    int ok = C.compile(code_text);
    if (ok < 0) return;
    for (int i = 0; i < MAX_EXEC_PROCESSES; i++) {
        if (!exec_contexts[i].active) {
            exec_contexts[i].active = true;
            exec_contexts[i].exec_id = i;
            exec_contexts[i].prog = C.pr;
            const char* av[] = {"exec", nullptr};
            exec_contexts[i].vm.start_execution(exec_contexts[i].prog,1,av,0,0,nullptr);
            return;
        }
    }
}
// =============================================================================
// Helper: write a short status string to VGA text mode row 1 (safe at any
// point in kernel_main, before framebuffer is initialised).
// =============================================================================
// Write msg to VGA text row (0-based)
static void vga_dbg_row(int row, const char* msg, uint8_t attr = 0x0F) {
    volatile uint16_t* vga = (volatile uint16_t*)0xB8000 + row * 80;
    int i = 0;
    for (; i < 79 && msg[i]; i++)
        vga[i] = (uint16_t)((uint16_t)(attr << 8) | (uint8_t)msg[i]);
    for (; i < 79; i++)
        vga[i] = (uint16_t)((uint16_t)(attr << 8) | ' ');
}

static void vga_status(const char* msg, uint8_t attr = 0x0F) {
    vga_dbg_row(1, msg, attr);
}

// Write a 32-bit hex value into buf[11] and return it
static char* hex32(uint32_t v, char* buf) {
    const char* h = "0123456789ABCDEF";
    buf[0]='0'; buf[1]='x';
    for (int i = 7; i >= 0; i--) { buf[2+i] = h[v & 0xF]; v >>= 4; }
    buf[10] = 0;
    return buf;
}

// Concatenate up to 6 strings into dst[128]
static char* vga_cat(char* dst, const char* a, const char* b="",
                      const char* c="", const char* d="") {
    char* p = dst;
    auto app = [&](const char* s){ while(*s && p < dst+127) *p++=*s++; };
    app(a); app(b); app(c); app(d);
    *p = 0; return dst;
}

// =============================================================================
// Framebuffer probe — four strategies, always returns something safe.
// =============================================================================
static void probe_framebuffer(multiboot_info* mbi,
                              uint32_t& fb_phys,
                              uint32_t& fb_w,
                              uint32_t& fb_h,
                              uint32_t& fb_pitch)
{
    fb_phys  = 0;
    fb_w     = 1024;
    fb_h     = 768;
    fb_pitch = 1024 * 4;

    // Diagnostic: show multiboot flags and framebuffer fields on VGA rows 2-4
    {
        char dbuf[128]; char hb[11];
        vga_dbg_row(2, vga_cat(dbuf, "MB flags=", hex32(mbi->flags, hb)), 0x0E);
        uint32_t raw_fb = (uint32_t)(uintptr_t)mbi->framebuffer_addr;
        char hb2[11];
        vga_dbg_row(3, vga_cat(dbuf, "FB addr=", hex32(raw_fb, hb),
                           " type=", hex32(mbi->framebuffer_type, hb2)), 0x0E);
        vga_dbg_row(4, vga_cat(dbuf, "FB w=", hex32(mbi->framebuffer_width, hb),
                           " h=", hex32(mbi->framebuffer_height, hb2)), 0x0E);
    }

    // Strategy 1: GRUB filled framebuffer fields (our boot.S requests this
    // via MB_FLAGS bit 2).  This is the normal path on QEMU + GRUB 2.
    // framebuffer_type: 0=indexed, 1=RGB/direct, 2=EGA text — only accept 1.
    if (mbi->flags & (1u << 12)) {
        uint32_t addr = (uint32_t)(uintptr_t)mbi->framebuffer_addr;
        char dbuf[128]; char hb[11];
        vga_dbg_row(5, vga_cat(dbuf, "S1: addr=", hex32(addr, hb),
                           " type=", hex32(mbi->framebuffer_type, hb)), 0x0A);
        // Accept type 1 (RGB direct) or type 0 (indexed/paletted reported by
        // some VMware GRUB configs). Reject type 2 (EGA text mode).
        if (addr >= 0x1000000u && mbi->framebuffer_type != 2) {
            fb_phys  = addr;
            fb_w     = mbi->framebuffer_width;
            fb_h     = mbi->framebuffer_height;
            fb_pitch = mbi->framebuffer_pitch;
            if (fb_w  > 1024) { fb_w  = 1024; fb_pitch = 1024*4; }
            if (fb_h  > 768)  { fb_h  = 768; }
            vga_dbg_row(5, "S1: SUCCESS - using GRUB framebuffer", 0x0A);
            return;
        }
        vga_dbg_row(5, "S1: SKIPPED (bad addr or type!=1)", 0x0E);
    } else {
        vga_dbg_row(5, "S1: SKIPPED (bit12 not set in flags)", 0x0E);
    }

    // Strategy 2: Bochs VBE ports (QEMU -vga std).
    {
        auto vbe_out = [](uint16_t idx, uint16_t val) {
            asm volatile("outw %0,%1" :: "a"(idx), "d"((uint16_t)0x01CE));
            asm volatile("outw %0,%1" :: "a"(val), "d"((uint16_t)0x01CF));
        };
        auto vbe_in = [](uint16_t idx) -> uint16_t {
            uint16_t v;
            asm volatile("outw %0,%1" :: "a"(idx), "d"((uint16_t)0x01CE));
            asm volatile("inw %1,%0"  : "=a"(v)   : "d"((uint16_t)0x01CF));
            return v;
        };
        vbe_out(0x04, 0x00);
        uint16_t id = vbe_in(0x00);
        if (id >= 0xB0C0) {
            vbe_out(0x01, 1024);
            vbe_out(0x02, 768);
            vbe_out(0x03, 32);
            vbe_out(0x05, 1024);
            vbe_out(0x06, 768);
            vbe_out(0x07, 0);
            vbe_out(0x08, 0);
            vbe_out(0x04, 0x41); // ENABLE | LFB_ENABLED
            for (uint16_t bus = 0; bus < 8 && !fb_phys; bus++) {
                for (uint8_t dev = 0; dev < 32 && !fb_phys; dev++) {
                    uint32_t vd = pci_read_config_dword(bus, dev, 0, 0x00);
                    if ((vd & 0xFFFF) == 0xFFFF) continue;
                    bool is_bochs   = (vd == 0x11111234u);
                    uint32_t cc     = pci_read_config_dword(bus, dev, 0, 0x08) >> 16;
                    bool is_display = (cc == 0x0300 || cc == 0x0380);
                    if (!is_bochs && !is_display) continue;
                    for (int b = 0; b < 3 && !fb_phys; b++) {
                        uint32_t bar = pci_read_config_dword(bus, dev, 0, 0x10 + b*4);
                        if (bar & 1) continue;
                        uint32_t addr = bar & 0xFFFFFFF0u;
                        if (addr >= 0x1000000u) fb_phys = addr;
                    }
                }
            }
            if (fb_phys) return;
        }
    }

    // Strategy 3: VMware SVGA II (vendor 0x15AD, device 0x0405).
    // The SVGA II adapter uses an I/O BAR (BAR0) for its index/value register
    // pair and a memory BAR (BAR1) for the linear framebuffer.  It must be
    // programmed via I/O ports to set the resolution and enable the FB;
    // simply reading BAR1 is not enough — the FB is not live until ENABLE=1.
    //
    // SVGA II register map (index written to io_base+0, value at io_base+1):
    //   SVGA_REG_ID       = 0   write SVGA_MAGIC|2 to negotiate version
    //   SVGA_REG_ENABLE   = 1   write 1 to enable SVGA mode
    //   SVGA_REG_WIDTH    = 2
    //   SVGA_REG_HEIGHT   = 3
    //   SVGA_REG_BPP      = 7   (bits per pixel)
    //   SVGA_REG_FB_START = 13  returns the physical FB base address
    //   SVGA_REG_PITCH    = 24  returns bytes per scan line
    {
        // Locate the SVGA II PCI device
        uint16_t svga_io = 0;
        uint32_t svga_fb_bar = 0;
        for (uint16_t bus = 0; bus < 8 && !svga_io; bus++) {
            for (uint8_t dev = 0; dev < 32 && !svga_io; dev++) {
                uint32_t vd = pci_read_config_dword(bus, dev, 0, 0x00);
                // VMware vendor 0x15AD, SVGA II device 0x0405
                if (vd != 0x040515ADu) continue;
                // BAR0 = I/O port base (bit 0 set = I/O space)
                uint32_t bar0 = pci_read_config_dword(bus, dev, 0, 0x10);
                if (bar0 & 1) svga_io = (uint16_t)(bar0 & 0xFFFE);
                // BAR1 = framebuffer memory base
                svga_fb_bar = pci_read_config_dword(bus, dev, 0, 0x14) & 0xFFFFFFF0u;

                // Enable PCI memory + I/O decode (command register, offset 4)
                uint32_t cmd = pci_read_config_dword(bus, dev, 0, 0x04);
                cmd |= 0x03; // I/O + Memory enable
                // pci_write_config_dword not available, use outl directly
                uint32_t addr_reg = 0x80000000u | ((uint32_t)bus << 16) |
                                    ((uint32_t)dev << 11) | 0x04u;
                outl(0xCF8, addr_reg);
                outl(0xCFC, cmd);
            }
        }

        {
            char dbuf[128]; char hb[11]; char hb2[11];
            vga_dbg_row(6, vga_cat(dbuf, "S3: io=", hex32(svga_io, hb),
                               " bar1=", hex32(svga_fb_bar, hb2)), 0x0A);
        }

        if (svga_io) {
            // Helper lambdas for SVGA II register access
            auto svga_write = [&](uint32_t reg, uint32_t val) {
                outl((uint16_t)(svga_io + 0), reg); // index port
                outl((uint16_t)(svga_io + 4), val); // value port
            };
            auto svga_read = [&](uint32_t reg) -> uint32_t {
                outl((uint16_t)(svga_io + 0), reg);
                return inl((uint16_t)(svga_io + 4));
            };

            // Negotiate SVGA II version (SVGA_ID_2 = 0x90000002)
            svga_write(0 /*SVGA_REG_ID*/, 0x90000002u);
            uint32_t id = svga_read(0);
            { char dbuf[128]; char hb[11];
              vga_dbg_row(7, vga_cat(dbuf, "S3: SVGA_ID=", hex32(id, hb)), 0x0A); }

            if (id == 0x90000002u) {
                // Set 1024x768x32
                svga_write(2 /*SVGA_REG_WIDTH*/,  1024);
                svga_write(3 /*SVGA_REG_HEIGHT*/,  768);
                svga_write(7 /*SVGA_REG_BITS_PER_PIXEL*/, 32);
                svga_write(1 /*SVGA_REG_ENABLE*/,  1);

                // Read back actual FB address and pitch
                uint32_t reported_fb = svga_read(13 /*SVGA_REG_FB_START*/);
                uint32_t pitch       = svga_read(24 /*SVGA_REG_BYTES_PER_LINE*/);

                { char dbuf[128]; char hb[11]; char hb2[11];
                  vga_dbg_row(8, vga_cat(dbuf, "S3: fb=", hex32(reported_fb, hb),
                                     " pitch=", hex32(pitch, hb2)), 0x0A); }

                // Use the reported address if valid, fall back to BAR1
                fb_phys  = (reported_fb >= 0x1000000u) ? reported_fb : svga_fb_bar;
                fb_w     = 1024;
                fb_h     = 768;
                fb_pitch = (pitch >= 1024*4) ? pitch : 1024*4;
                if (fb_phys >= 0x1000000u) {
                    vga_dbg_row(9, "S3: SUCCESS - VMware SVGA II programmed", 0x0A);
                    return;
                }
                vga_dbg_row(9, "S3: FAILED - fb addr still bad", 0x4F);
            } else {
                vga_dbg_row(7, "S3: FAILED - wrong SVGA_ID (not VMware?)", 0x4F);
            }
        } else {
            vga_dbg_row(6, "S3: SKIPPED - no SVGA II device found on PCI", 0x0E);
        }
    }

    // Strategy 4: hardcoded fallback.
    // 0xFD000000 = Bochs/QEMU default.
    // 0xE8000000 = VMware Workstation SVGA II BAR1 (confirmed from vmware.log).
    // vmware_svga_init() runs before probe_framebuffer and will override this
    // with the correct BAR1 address, so this fallback is only for QEMU/Bochs.
    fb_phys = 0xFD000000u;
}

// =============================================================================
// kernel_main
// =============================================================================
// ─────────────────────────────────────────────────────────────────────────────
// VMware SVGA II initialisation.
// Must run BEFORE any framebuffer access.  Returns the live FB base address,
// or 0 if no SVGA II device is found.
//
// The SVGA II I/O port pair lives at BAR0 (an I/O BAR):
//   index port = BAR0 + 0
//   value port = BAR0 + 4        (NOT +1 — the value port is 32-bit wide)
// Register indices used here:
//   0  SVGA_REG_ID              write 0x90000002 to negotiate SVGA2
//   1  SVGA_REG_ENABLE          write 1 to enable linear framebuffer
//   2  SVGA_REG_WIDTH
//   3  SVGA_REG_HEIGHT
//   7  SVGA_REG_BITS_PER_PIXEL
//  13  SVGA_REG_FB_START        read to get physical FB address
//  24  SVGA_REG_BYTES_PER_LINE  read to get pitch in bytes
// ─────────────────────────────────────────────────────────────────────────────
struct SVGAResult { uint32_t fb; uint32_t pitch; bool ok; };

// ── VMware SVGA II: full PCI scan + I/O programming ──────────────────────────
// Scans ALL 256 PCI buses (some VMware configs place SVGA on bus > 7),
// all 32 devices, all 8 functions.  Tries both device IDs 0x0405 and 0x0710.
// The I/O BAR may be at BAR0 or BAR2 depending on SVGA revision.
static SVGAResult vmware_svga_init(uint32_t w, uint32_t h) {
    SVGAResult r = {0, w*4, false};

    uint16_t io   = 0;
    uint32_t bar1 = 0;

    // Full PCI scan — VMware may put the SVGA on any bus
    for (uint32_t bus = 0; bus < 256 && !io; bus++) {
        for (uint32_t dev = 0; dev < 32 && !io; dev++) {
            for (uint32_t fn = 0; fn < 8 && !io; fn++) {
                uint32_t id = pci_read_config_dword(
                    (uint16_t)bus, (uint8_t)dev, (uint8_t)fn, 0x00);
                if ((id & 0xFFFFu) != 0x15ADu) continue; // not VMware vendor
                uint32_t did = (id >> 16) & 0xFFFFu;
                if (did != 0x0405u && did != 0x0710u) continue; // not SVGA

                // DO NOT touch the PCI command register.
                // The BIOS has already enabled I/O + Memory decode for SVGA.
                // Re-writing the command register causes VMware to briefly
                // unmap BAR1 (0xe8000000), creating a window where pixel
                // writes crash with "execute an invalid part of memory".

                // Read all 6 BARs — find the I/O BAR and the memory BAR
                for (int b = 0; b < 6; b++) {
                    uint32_t bar = pci_read_config_dword(
                        (uint16_t)bus, (uint8_t)dev, (uint8_t)fn,
                        (uint8_t)(0x10 + b*4));
                    if ((bar & 1u) && !io) {
                        io = (uint16_t)(bar & 0xFFFCu); // I/O BAR
                    } else if (!(bar & 1u) && !bar1) {
                        uint32_t addr = bar & 0xFFFFFFF0u;
                        if (addr >= 0x1000000u) bar1 = addr; // memory BAR
                    }
                }
            }
        }
    }

    // Also try the fixed legacy SVGA I/O port (0x4560) used by very old VMware
    if (!io) io = 0x4560u;

    // I/O helpers
    auto wr = [&](uint32_t reg, uint32_t val) {
        outl(io,     reg);
        outl(io + 4, val);
    };
    auto rd = [&](uint32_t reg) -> uint32_t {
        outl(io, reg);
        return inl(io + 4);
    };

    // Negotiate SVGA2 — try SVGA_ID_2, fall back to SVGA_ID_1
    wr(0, 0x90000002u);
    uint32_t svga_id = rd(0);
    if (svga_id != 0x90000002u) {
        wr(0, 0x90000001u);
        svga_id = rd(0);
        if (svga_id != 0x90000001u) return r; // not responding
    }

    // Program resolution
    wr(2, w);
    wr(3, h);
    wr(7, 32);

    // Enable
    wr(1, 1);

    // Read FB address and pitch.
    // VMware Workstation 14+ returns 0 from SVGA_REG_FB_START (reg 13) —
    // the physical framebuffer is always at BAR1. Prefer BAR1 when valid.
    uint32_t fb    = rd(13);
    uint32_t pitch = rd(24);

    if (bar1 >= 0x1000000u) fb = bar1;  // BAR1 is authoritative on VMware
    else if (fb < 0x1000000u) fb = bar1;
    if (pitch < w * 4) pitch = w * 4;

    r.fb    = fb;
    r.pitch = pitch;
    r.ok    = (fb >= 0x1000000u);
    return r;
}


extern "C" void kernel_main(uint32_t magic, uint32_t multiboot_addr) {

    // ── Verify Multiboot 1 magic FIRST, before any hardware probing ───────────
    // If GRUB (or whatever bootloader) didn't pass 0x2BADB002, we are running
    // under something that doesn't honour the contract — bail out cleanly
    // instead of poking PCI / FB hardware on bad assumptions.
    if (magic != 0x2BADB002) {
        volatile uint16_t* vga = (volatile uint16_t*)0xB8000;
        vga[0] = 0x4F45; // 'E' on red — "bad magic"
        for (;;)
            asm volatile("hlt");
    }

    // ── Initialise heap ───────────────────────────────────────────────────────
    g_allocator.init(kernel_heap, sizeof(kernel_heap));

    // ── Step 1: unconditionally try VMware SVGA II first. ─────────────────────
    // This MUST happen before reading any framebuffer address because the
    // linear FB is not live until ENABLE=1 is written.
    SVGAResult svga = vmware_svga_init(1024, 768);

    // ── Step 2: determine framebuffer address ────────────────────────────────
    multiboot_info* mbi = (multiboot_info*)multiboot_addr;

    if (svga.ok) {
        // VMware SVGA II programmed successfully — use its reported address.
        // Wait for VMware to complete the MemSpace re-registration after ENABLE=1.
        // The log shows VMware briefly unmaps/remaps 0xe8000000 during SVGA init;
        // a short spin ensures the mapping is live before we write pixels.
        for (volatile uint32_t i = 0; i < 5000000u; i++);
		
        fb_info = { (uint32_t*)svga.fb, 1024, 768, svga.pitch };
    } else {
        // Not VMware (or SVGA II failed) — use GRUB multiboot info directly.
        // Accept any type except 2 (EGA text). If bit 12 not set, fall back
        // to probing Bochs VBE ports, then hardcoded candidates.
        uint32_t fb_phys = 0, fb_w = 1024, fb_h = 768, fb_pitch = 1024*4;

        if ((mbi->flags & (1u << 12)) && mbi->framebuffer_type != 2) {
            // GRUB filled the framebuffer fields (because boot.S requested
            // video mode via Multiboot1 FLAGS bit 2). Take its values
            // verbatim; this is the normal path on real BIOS + VMware-BIOS
            // and on UEFI through GRUB-EFI (which sets it up via GOP).
            fb_phys  = (uint32_t)(uintptr_t)mbi->framebuffer_addr;
            fb_w     = mbi->framebuffer_width  ? mbi->framebuffer_width  : 1024;
            fb_h     = mbi->framebuffer_height ? mbi->framebuffer_height : 768;
            fb_pitch = mbi->framebuffer_pitch  ? mbi->framebuffer_pitch  : fb_w*4;
        }

        if (fb_phys < 0x1000000u) {
            // GRUB gave nothing useful — try Bochs VBE then hardcoded
            probe_framebuffer(mbi, fb_phys, fb_w, fb_h, fb_pitch);
        }

        fb_info = { (uint32_t*)(uintptr_t)fb_phys, fb_w, fb_h, fb_pitch };
    }

    if (fb_info.width  == 0 || fb_info.width  > 1920) fb_info.width  = 1024;
    if (fb_info.height == 0 || fb_info.height > 1200) fb_info.height = 768;
    if (fb_info.pitch  < fb_info.width * 4) fb_info.pitch = fb_info.width * 4;

    // ── Step 3: commit and paint ──────────────────────────────────────────────
    backbuffer = backbuffer_storage;
    g_gfx.init(false);

    g_gfx.clear_screen(ColorPalette::DESKTOP_GRAY );
    swap_buffers();

    // ── Open first terminal window ────────────────────────────────────────────
    launch_new_terminal();

    // ── PS/2 mouse ────────────────────────────────────────────────────────────
    ps2_flush_output_buffer();
    if (initialize_universal_mouse()) {
        wm.print_to_focused("Mouse: initialised.\n");
    } else {
        wm.print_to_focused("Mouse: init failed (keyboard-only mode).\n");
    }

    // ── AHCI disk + FAT32 ─────────────────────────────────────────────────────
    disk_init();
    // Don't hardcode a port selection here — disk_init() already
    // auto-selected the first port with an attached device. Hardcoding
    // "1" only worked on the QEMU command line in compile.md (which puts
    // the hard disk on ahci.1); bare-metal and VMware almost always have
    // the boot disk on port 0 and would have silently fallen back to no
    // selection. Pass "" to just list the detected ports for the user.
    cmd_list_and_select_disk("");

    if (ahci_base) {
        fat32_init();
        wm.print_to_focused("Disk: AHCI found.\n");
    } else {
        wm.print_to_focused("Disk: no AHCI controller.\n");
    }
    if (current_directory_cluster) {
        wm.load_desktop_items();
        if (extract_busybox_to_filesystem())
            wm.print_to_focused("BusyBox: saved to FAT32.\n");
        else
            wm.print_to_focused("BusyBox: ramdisk empty or write failed.\n");
        if (extract_hello_to_filesystem())
            wm.print_to_focused("hello: saved to FAT32.\n");

        // Write the TCC guest ABI header so user programs can #include "tcc.h"
        // to get outb / kprint / kexit without redefining them.
        {
            fat_dir_entry_t tcc_hdr_entry;
            uint32_t th_sec = 0, th_off = 0;
            if (fat32_find_entry("tcc.h", &tcc_hdr_entry, &th_sec, &th_off) != 0) {
                static const char tcc_h[] =
                    "/* tcc.h — guest ABI for in-kernel TCC programs */\n"
                    "#ifndef TCC_H\n#define TCC_H\n"
                    "static inline void outb(unsigned short p, unsigned char v) {\n"
                    "    __asm__ volatile(\"outb %0,%1\"::\"a\"(v),\"Nd\"(p)); }\n"
                    "static inline void kprint(const char* s) {\n"
                    "    while (*s) outb(0xE9, *s++); }\n"
                    "static inline void kexit(int c) {\n"
                    "    outb(0xE8, (unsigned char)c); }\n"
                    "#endif\n";
                fat32_write_file("tcc.h", tcc_h, (uint32_t)(sizeof(tcc_h) - 1));
            }
        }
    } else {
        wm.print_to_focused("FAT32: not initialised.\n");
    }

    // ── Bochs CPU / ELF subsystem ─────────────────────────────────────────────
    //
    // Run the file-scope C++ constructors NOW, before init_elf_system()
    // and before any code can reach a Bochs entry point. boot.S does not
    // walk __init_array, so without this the Bochs core objects (bx_cpu,
    // bx_mem, the CPUID param objects, icache's pageWriteStampTable, ...)
    // stay as zero-filled BSS with null vtables. The first ELF launched
    // in the Bochs emulator window would then call bochs_cpu_init() ->
    // BX_CPU(0)->initialize() against those null vtables and fault out on
    // its very first tick — the "first-time crash / autoclose" symptom.
    kernel_run_global_ctors_once();

    // The constructors have now run exactly once. Tell the test module so
    // its own run_init_array_once() inside test_module_run() becomes a
    // no-op — otherwise the first `test` command would re-run every ctor
    // a second time, re-constructing bx_cpu / bx_mem on top of live
    // (already-initialised) Bochs state.
    test_module_mark_ctors_done();

    init_elf_system();

    vga_status("Init complete - entering main loop", 0x0A);

    // ── Main loop ─────────────────────────────────────────────────────────────
    uint32_t last_paint_tick = 0;
    const uint32_t TICKS_PER_FRAME = 1;
    int prev_mouse_x = mouse_x;
    int prev_mouse_y = mouse_y;

    // Force an immediate first render — don't wait 500 poll iterations
    // for the software timer to tick before the desktop appears.
    g_evt_timer = true;
    g_evt_dirty = true;

    // Spinning heartbeat at VGA column 79, row 0 (green)
    volatile uint16_t* vga_hb = (volatile uint16_t*)(0xB8000 + 2*79);
    uint32_t hb_counter = 0;
    const char hb_chars[] = "|/-\\";
    static uint32_t poll_counter = 0;

    for (;;) {
        if (++hb_counter % 10000 == 0) {
            *vga_hb = (uint16_t)(0x0A00u | (uint8_t)hb_chars[(hb_counter/10000)%4]);

		}


        bool prev_left  = mouse_left_down;
        bool prev_right = mouse_right_down;
        poll_input_universal();

        bool leftClickedThisFrame  = (mouse_left_down  && !prev_left);
        bool rightClickedThisFrame = (mouse_right_down && !prev_right);
        bool mouse_moved = (mouse_x != prev_mouse_x || mouse_y != prev_mouse_y);
        bool key_pressed = (last_key_press != 0);

        if (key_pressed || mouse_moved || leftClickedThisFrame || rightClickedThisFrame) {
            g_evt_input = true;
            g_input_state.hasNewInput = true;
            prev_mouse_x = mouse_x;
            prev_mouse_y = mouse_y;
        }

        // Route keypresses to any active ELF guest process
        //
        // FIX (focus ignored on click): this used to scan every slot
        // and hand the keystroke to the FIRST one that was active &&
        // waiting_for_input, regardless of which terminal window was
        // actually focused/clicked. With two terminals each running
        // a program that reads stdin, typing while Terminal B was
        // focused would silently feed Terminal A instead (whichever
        // slot happened to be waiting), and last_key_press got
        // zeroed here before wm.handle_input ever saw it — so B's
        // own captured_elf_slot path (kernel.cpp's BUSYBOX CAPTURE
        // block) never even ran.
        //
        // Fix: only steal the keystroke for the ELF slot owned by
        // the currently FOCUSED window. If that slot isn't waiting
        // for input (or no window is focused, or the focused window
        // isn't capturing a slot), fall through and let the normal
        // g_evt_input / wm.handle_input path below handle the key —
        // which is what already correctly threads input to whichever
        // terminal's captured_elf_slot the user clicked into.
        if (last_key_press != 0) {
            int fs = wm.get_focused_elf_slot();
            if (fs >= 0 && fs < MAX_ELF_PROCESSES &&
                elf_processes[fs].active && elf_processes[fs].waiting_for_input) {
                push_input(fs, last_key_press);
                elf_processes[fs].waiting_for_input = false;
                last_key_press = 0;
            }
        }

        // Software timer (no PIT — IRQ0 would fire into an unhandled vector)
        if (++poll_counter >= 500) {
            poll_counter  = 0;
            g_evt_timer   = true;
			g_evt_dirty = true;
            g_timer_ticks++;
        }

        if (g_evt_input) {
            g_evt_input = false;
            wm.handle_input(last_key_press, mouse_x, mouse_y,
                            mouse_left_down,
                            leftClickedThisFrame,
                            rightClickedThisFrame);
            if (last_key_press != 0) last_key_press = 0;
            g_evt_dirty = true;
        }

        wm.cleanup_closed_windows();

        if (g_evt_timer && (g_timer_ticks - last_paint_tick) >= TICKS_PER_FRAME) {
            // Tick ELF processes BEFORE the paint so any breadcrumbs they
            // write (x86_breadcrumb at row 2, glue's tick markers at row 0
            // col 72/73, panic tags at col 70) are reflected in the next
            // swap_buffers. Otherwise a hang inside tick_elf_processes
            // would leave the last painted frame without the breadcrumbs
            // pointing at where the hang happened.
            tick_elf_processes(100);

            if (g_evt_dirty || g_input_state.hasNewInput) {
                last_paint_tick           = g_timer_ticks;
                g_evt_dirty               = false;
                g_input_state.hasNewInput = false;
                g_gfx.clear_screen(ColorPalette::DESKTOP_GRAY );
                wm.update_all();
                draw_cursor(mouse_x, mouse_y, ColorPalette::CURSOR_WHITE);
                // Diagnostic overlay: paint VGA text-mode rows 0/1/2
                // (boot/panic/tick breadcrumbs, host-IDT fault tags,
                // x86_tick lazy-init progress) onto the framebuffer so
                // they are visible in graphics mode. Drawn last so it
                // overlays everything.
                draw_vga_overlay();
                swap_buffers();
            }

            g_evt_timer = false;
        }
    }
}
// =============================================================================
// extern "C" bridges for tcc_kernel.cpp
// =============================================================================
// tcc_kernel.cpp is compiled as freestanding C++ and cannot include the full
// kernel headers. These thin wrappers expose the symbols it needs with plain
// C linkage so the linker resolves them without name-mangling.

extern "C" {

void tcc_bridge_console_print(const char* s) {
    console_print(s);
}
static inline bool is_cc_safe_char(unsigned char c) {
    return c == '\n' || c == '\r' || c == '\t' || (c >= 32 && c != 127);
}

char* tcc_bridge_fat32_read(const char* filename) {
    char *data = fat32_read_file_as_string(filename);

    if (!data) {
        return NULL;
    }

    for (char *p = data; *p; ++p) {
        if (!is_cc_safe_char((unsigned char)*p)) {
            
            return NULL;
        }
    }

    return data;
}

int tcc_bridge_fat32_write(const char* filename, const void* data, unsigned int size) {
    return fat32_write_file(filename, data, (uint32_t)size);
}

int tcc_bridge_exec_elf(void* terminal, const char* filename, const char* args) {
    TerminalWindow* tw = (TerminalWindow*)terminal;
    return tw->exec_elf(filename, args);
}

} // extern "C"