#include <stdio.h>
#include <bitset>
#include <cmath>

const int BASE = 2;
const int BIT_COUNT = 32;

float derCorput(unsigned i)
{
	std::bitset<BIT_COUNT> bits(i);
	float total = 0;
	int power = -1;

	// When iterating, bits are already in reversed order
	for (int i = 0; i < BIT_COUNT; ++i)
		total += (bits[i] * pow(BASE, power--));

	return total;
}

int main(int argc, char* argv[])
{
	for (auto i = 0; i < 20; ++i)
		printf("%f, ", derCorput(i));

	return 0;
}