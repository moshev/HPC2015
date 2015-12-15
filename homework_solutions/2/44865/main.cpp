#include <iostream>
#include <algorithm>
#include <mmintrin.h>

const unsigned char CHUNK_SIZE = 64;

void reverse(char* bytes, int numChunks) {
	std::reverse(bytes, bytes + numChunks * CHUNK_SIZE);
    char* arr = bytes;
	//char* arr = (char*)_mm_malloc(numChunks * CHUNK_SIZE, 1);

	for (int i = 0, j = numChunks * CHUNK_SIZE - CHUNK_SIZE, end = numChunks * CHUNK_SIZE; 
	i < numChunks * CHUNK_SIZE;
		i += CHUNK_SIZE, j -= CHUNK_SIZE, end -= CHUNK_SIZE) {
		std::move(&bytes[j], &bytes[end], &arr[i]);
	}

	std::move(arr, &arr[numChunks * CHUNK_SIZE], bytes);

	//_mm_free(arr);
}
