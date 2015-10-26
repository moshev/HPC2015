#include <iostream>
#include <cmath>

using namespace std;

float derCorput(unsigned n) {
	float result = 0;
	int k = 0; // or maybe unsigned?

	while(n != 0) {
		const unsigned temp = n;
		const short d = temp % 2;
		const int tmpK = k; 

		n = temp >> 1;
		result += d*pow(2.0f,(-tmpK-1));
		k = tmpK + 1;
	}

	return result;
}

int main() {
	for (auto i = 0; i < 20; ++i)
		printf("%f, ", derCorput(i));

	return 0;
}