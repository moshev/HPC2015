#include <iostream>
#include <vector>
#include <cmath>

using std::vector;

vector<bool> binary(unsigned n) {
	vector<bool> binaryN;

	if (n == 0)
		return binaryN;
	else {
		while (n > 1) {
            n % 2 == 0 ? binaryN.push_back(0) : binaryN.push_back(1);
			n /= 2;
		}

		binaryN.push_back(1);
		return binaryN;
	}
}

float derCorput(unsigned n) {
	vector<bool> binaryN = binary(n);

	float sum = 0.0;

	for (int i = 0, index = 0; index < binaryN.size(); ++i, ++index) {
		sum += (binaryN[index] * static_cast<float>(pow(2.0, (- i - 1))));
	}

	return sum;
}

int main() {

}
