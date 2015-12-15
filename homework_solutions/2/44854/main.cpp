#include "tmmintrin.h"
#include "immintrin.h"
#include <cstdio>


#define CHUNK_SIZE 64
#define CHUNK_PART_SIZE 16
#define CHUNK_PARTS_COUNT 4


//using https://software.intel.com/sites/landingpage/IntrinsicsGuide/
//and further explanations here - http://stackoverflow.com/questions/12778620/mm-shuffle-epi8-and-an-example
void reverse(char* bytes, int numChunks)
{
	//64 bytes as an array of 4 16-byte parts
	__m128i chunkParts[CHUNK_PARTS_COUNT];

	//set the mask so that all the bytes in each 16-byte part in the 64-byte chunk is reversed
	__m128i shuffleControlMask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);


	int numBytes = numChunks * CHUNK_SIZE;
	for(int i = 0; i < numBytes; i += CHUNK_SIZE)
	{
		//load current chunk
		for(int j = 0; j < CHUNK_PARTS_COUNT; ++j)
			chunkParts[j] = _mm_loadu_si128((__m128i *)&bytes[i + CHUNK_PART_SIZE*j]);

		//shuffle bytes in each part of the chunk so that they're reversed and
		//then store/write each chunk part back in the original array ordering them correctly
		for(int j = 0; j < CHUNK_PARTS_COUNT; ++j)
			_mm_storeu_si128((__m128i *)&bytes[i + (CHUNK_PARTS_COUNT - 1 - j)*CHUNK_PART_SIZE],
							 _mm_shuffle_epi8(chunkParts[j] ,shuffleControlMask));
	}


}


int main()
{
	return 0;
}
