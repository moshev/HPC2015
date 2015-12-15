#include <stdio.h>
//#include <chrono>
//#include <iostream>
//using namespace std::chrono;

#define CHUNK_SIZE 64

void reverse(char *bytes, int numChunks)
{
	char swap_value;
	for (int chunk_num = 0; chunk_num < numChunks; ++chunk_num)
	{
		//printf("\n%d\n", chunk_num);
		for (int chunk_index = 0; chunk_index <  CHUNK_SIZE/2; ++chunk_index)
		{
			//printf("%d ", chunk_index);
			//printf("swap %d %d\n", chunk_num * CHUNK_SIZE + chunk_index,  chunk_num * CHUNK_SIZE + CHUNK_SIZE - 1 - chunk_index);
			swap_value = bytes[chunk_num * CHUNK_SIZE + chunk_index];
			bytes[chunk_num * CHUNK_SIZE + chunk_index] = bytes[chunk_num * CHUNK_SIZE + CHUNK_SIZE - 1 - chunk_index];
			bytes[chunk_num * CHUNK_SIZE + CHUNK_SIZE - 1 - chunk_index] = swap_value; 
		}
	}
}

void test_reverse(int size)
{
    char bytes[256];
    for (int i = 0; i < size; ++i)
    {
        bytes[i] = i;
    }
    //auto start = std::chrono::system_clock::now();
    reverse(bytes, (size/64));
    /*
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << diff.count() << std::endl;
    */
    //printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]);
}

int main(int argc, char* argv[])
{
	test_reverse(128);
	return 0;
}
