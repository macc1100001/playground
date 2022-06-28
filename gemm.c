#include<stdio.h>
#include<stdint.h>
#include<time.h>
#include<assert.h>
#include<omp.h>

#define N 4096

float A[N*N];
float B[N*N];
float C[N*N];

uint64_t nanos(){
	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

#define BLOCK 4

void mmult(){
#pragma omp parallel
	{
	//#pragma omp for
	for(int by = 0; by < N; by += BLOCK){
		#pragma omp for
		for(int bx = 0; bx < N; bx += BLOCK){
			float tc[BLOCK*BLOCK] = {};
			for(int k = 0; k < N; ++k){
				for(int y = 0; y < BLOCK; ++y){
					for(int x = 0; x < BLOCK; ++x){
						 tc[y*BLOCK + x] += A[(by+y)*N +k] * B[(bx+x)*N + k];
					}
				}
			}
			for(int y = 0; y < BLOCK; ++y){
				for(int x = 0; x < BLOCK; ++x){
					C[(by+y)*N + bx+x] = tc[y*BLOCK + x];
				}
			}
		}
	}

	}
}

int main(){
	uint64_t start = nanos();
	
	mmult();
	uint64_t end = nanos();
	double gflop = (2.0*N*N*N)*1e-9;
	double s = (end-start)*1e-9;
	printf("%f GFLOP\n", gflop);
	printf("%f GFLOP/S\n", gflop/s);
	return 0;
}

