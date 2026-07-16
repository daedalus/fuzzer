/* Simple protocol parser — easier than PNG but real branching.
 * Magic sequence: "OPEN" unlocks deeper paths.
 * Various crash conditions at different depths. */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

int main(void) {
    uint8_t buf[512];
    ssize_t n = read(0, buf, sizeof(buf) - 1);
    if (n < 4) return 0;
    buf[n] = '\0';

    /* Path 1: magic header check */
    if (buf[0] == 'O' && buf[1] == 'P' && buf[2] == 'E' && buf[3] == 'N') {
        /* Path 2: version check */
        if (n >= 5 && buf[4] == 'V') {
            /* Path 3: command dispatch */
            if (n >= 6) {
                switch (buf[5]) {
                    case 'R':  /* READ command */
                        if (n >= 8 && buf[6] == 'L' && buf[7] == 'E') {
                            /* path: read leading edge */
                            if (n >= 10) {
                                uint16_t offset = *(uint16_t*)(buf + 8);
                                if (offset == 0xDEAD) {
                                    /* crash: null deref */
                                    char *p = NULL;
                                    *p = 42;
                                }
                            }
                        }
                        break;
                    case 'W':  /* WRITE command */
                        if (n >= 7 && buf[6] == 'X') {
                            /* crash: heap overflow */
                            char *p = malloc(8);
                            memset(p, 'A', 64);
                            free(p);
                        }
                        break;
                    case 'D':  /* DELETE command */
                        if (n >= 9 && buf[6] == 'E' && buf[7] == 'L'
                            && buf[8] == 'E') {
                            /* crash: double free */
                            char *p = malloc(16);
                            free(p);
                            free(p);
                        }
                        break;
                    case 'S':  /* STATUS command */
                        if (n >= 8 && buf[6] == 'U' && buf[7] == 'M') {
                            /* deep path: checksum verification */
                            if (n >= 12) {
                                uint32_t cs = *(uint32_t*)(buf + 8);
                                if (cs == 0xCAFEBABE) {
                                    /* crash: stack overflow */
                                    volatile char big[4];
                                    memset((char*)big, 'Z', 256);
                                }
                            }
                        }
                        break;
                }
            }
        } else if (n >= 5 && buf[4] == 'X') {
            /* alternate path: extended mode */
            if (n >= 6 && buf[5] == 'P') {
                /* crash: use after free */
                char *p = malloc(32);
                free(p);
                volatile char c = p[0];
                (void)c;
            }
        }
    } else if (buf[0] == 'C' && buf[1] == 'L' && buf[2] == 'O' && buf[3] == 'S') {
        /* CLOSE path */
        if (n >= 6 && buf[4] == 'E' && buf[5] == 'D') {
            /* crash: abort */
            abort();
        }
    }
    return 0;
}
