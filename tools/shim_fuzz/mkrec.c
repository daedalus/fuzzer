#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/ipc.h>
int main(void){ size_t sz=16384; int id=shmget(IPC_PRIVATE,sz,IPC_CREAT|0600); if(id<0){perror("shmget");return 1;}
 uint64_t *p=shmat(id,0,0); memset(p,0,sz); p[0]=0x3230434552504d43ULL; uint32_t *w=(uint32_t*)p;
 w[2]=1; w[3]=64; w[4]=4; w[5]=64; w[6]=242; shmdt(p); printf("%d\n",id); return 0; }
