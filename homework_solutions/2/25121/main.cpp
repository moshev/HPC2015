#include <cstring>
#include "immintrin.h"

//#include <cstdio>
//char input[64] = {12,21,23,32,34,43,45,54,56,65,67,76,78,87,89,98,
//                  22,31,33,42,44,53,55,64,66,75,77,16,38,47,19,68,
//                  32,41,43,52,54,63,65,74,76,85,87,26,28,57,29,58,
//                  42,51,53,62,64,73,75,84,86,95,97,36,18,67,39,48};
char mask[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};

void reverse(char* bytes, int numChunks) {
	__m128i a[4],b,result;
	memcpy((char*)&b, (char*)mask, 16);
	for(int chunk=0; chunk<numChunks; ++chunk) {
		memcpy((char*)a, bytes+chunk*64, 64);
		for(int i=0; i<4; ++i) {
			result = _mm_shuffle_epi8 (a[i], b);
			memcpy((char*) (bytes+(3-i)*16+chunk*64), (char*)&result, 16);
		}
	}
}

//int main() {
//	char bytes[128];
//	for (int i = 0; i < 128; ++i)
//	   bytes[i] = i;
//	
//	reverse(bytes, (128/64));
//	
//	printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]); //63 62 65 64
//}