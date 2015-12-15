#include <stdio.h>

const int CHUNK_SIZE = 64;

void reverse(char* bytes, int numChunks)
{
	char* low = bytes;
	char* high = bytes + CHUNK_SIZE - 1;
	char swap = 0;

	for (int i = 0; i < numChunks; ++i)
	{
		while (low < high)
		{
			swap = *low;
			*low++ = *high;
			*high-- = swap;
		}

		low += (CHUNK_SIZE / 2);
		high = low + CHUNK_SIZE - 1;
	}
}

int main()
{
	char bytes[128];
	for (int i = 0; i < 128; ++i)
		bytes[i] = i;

	reverse(bytes, (128 / 64));

	printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]); //63 62 65 64
}