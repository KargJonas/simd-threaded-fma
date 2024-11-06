## SIMD + Multithreading + FMA in C

This is a little experiment to see what kinds of throughputs I can achieve for parallelized fused mulacc one a CPU with AVX2 and multithreading.
As it stands now, the hand-crafted implementation is roughly 20% faster than the compiler optimized naive implementation.

Build and run naive impl:
```bash
 gcc -O3 -fopenmp -mavx2 src/naive.c -o build/native  && ./build/native
```


Build hand crafted impl:
```bash
make main && ./build/main
```