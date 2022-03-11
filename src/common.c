#include "common.h"

LOAD LOAD_SELECT;
COMPUTE COMPUTE_SELECT;
COMPARE COMPARE_SELECT;

uint32_t m;
uint32_t k;
uint32_t n;

uint64_t X[M][K];
uint64_t Y[K][N];
uint64_t YP[N][K];
uint64_t *Xp;
uint64_t *Yp;

uint32_t X32[M][K];
uint32_t Y32[K][N];
uint32_t YP32[N][K];
uint32_t *Xp32;
uint32_t *Yp32;

uint16_t X16[M][K];
uint16_t Y16[K][N];
uint16_t YP16[N][K];
uint16_t *Xp16;
uint16_t *Yp16;

uint64_t Z[M][N]; // target
uint64_t ZP[N][M]; // target
uint64_t *Zp; // target
uint64_t R[M][N]; // reference
