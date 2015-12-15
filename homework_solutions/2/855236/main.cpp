#include <iostream>

void swapp (int& num1, int& num2) // used for swaping two elements
{
	int tmp;
	tmp = num1;
	num1=num2;
	num2 = tmp;
}

void reverse(char*array2, int numChunks)
{
    int* array = (int*)array2;
	size_t decr = 63;			//used for getting elements from right to left  
	for (int i = 0; i < numChunks;++i)
	{
		
		for(int j = 0; j< 32; j++)
		{
			swapp(array[(i*64)+j], array[(i*64)+(decr-j)] );
		}
	}
}
