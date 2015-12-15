#include<cstdio>
#include<emmintrin.h>

union UU
{
	char data[16];
	__m128i vec;
};

void reverse(char* bytes, int numChunks)
{
    // numChunks    == countof(bytes) / 64
    // countof(arr) == countof(bytes) / 16
	UU* arr = (UU*)bytes;
	UU  tmp0, tmp1, tmp2, tmp3;
	for (int i = 0; i < numChunks; i++)
	{
	    i<<=6;

		// 4 separate, truly independent (as in no-dependencies) calculations:
		tmp0.vec = _mm_set_epi8(bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3], bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7],
								bytes[i + 8], bytes[i + 9], bytes[i + 10], bytes[i + 11], bytes[i + 12], bytes[i + 13], bytes[i + 14], bytes[i + 15]);
		
		tmp1.vec = _mm_set_epi8(bytes[i + 16], bytes[i + 17], bytes[i + 18], bytes[i + 19], bytes[i + 20], bytes[i + 21], bytes[i + 22], bytes[i + 23],
								bytes[i + 24], bytes[i + 25], bytes[i + 26], bytes[i + 27], bytes[i + 28], bytes[i + 29], bytes[i + 30], bytes[i + 31]);
		
		tmp2.vec = _mm_set_epi8(bytes[i + 32], bytes[i + 33], bytes[i + 34], bytes[i + 35], bytes[i + 36], bytes[i + 37], bytes[i + 38], bytes[i + 39],
								bytes[i + 40], bytes[i + 41], bytes[i + 42], bytes[i + 43], bytes[i + 44], bytes[i + 45], bytes[i + 46], bytes[i + 47]);
		
		tmp3.vec = _mm_set_epi8(bytes[i + 48], bytes[i + 49], bytes[i + 50], bytes[i + 51], bytes[i + 52], bytes[i + 53], bytes[i + 54], bytes[i + 55],
								bytes[i + 56], bytes[i + 57], bytes[i + 58], bytes[i + 59], bytes[i + 60], bytes[i + 61], bytes[i + 62], bytes[i + 63]);
		
		i >>= 4;
		
		// 4 more independent writes, probably unnecessary...
		arr[i].vec = tmp3.vec;
		arr[i + 1].vec = tmp2.vec;
		arr[i + 2].vec = tmp1.vec;
		arr[i + 3].vec = tmp0.vec;
		
		i >>= 2;
	}
}

int main()
{
	char bytes[128];
	for (int i = 0; i < 128; i++) bytes[i] = i;

	reverse(bytes, 128 / 64);

	//for (int i = 0; i < 128; i++) printf("%3i%c",int(bytes[i]),(i + 1 & 15) ? ' ' : '\n');
	printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]);
}