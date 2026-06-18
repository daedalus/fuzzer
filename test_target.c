#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    char buf[64];
    int n = read(0, buf, sizeof(buf));
    if (n < 4) return 0;
    if (buf[0] == 'H' && buf[1] == 'T' && buf[2] == 'T' && buf[3] == 'P')
        abort();
    return 0;
}
