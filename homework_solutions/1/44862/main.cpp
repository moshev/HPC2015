#include <iostream>

using namespace std;

void fillArray(float* arr)
{
    int size = sizeof(unsigned) * 8;
    float multiplier = 0.5;

    arr[0] = multiplier;
    for(int i = 1; i < size; ++i) {
        arr[i] = arr[i - 1] * multiplier;
    }
}

float derCorput(unsigned n)
{
    float res = 0;
    static float multipliers[sizeof(unsigned) * 8];
    static bool initialized = false;
    if(!initialized) {
        fillArray(multipliers);
        initialized = true;
    }
    for(int i = 0; n != 0; ++i) {
        res += (n & 1) * multipliers[i];
        n >>= 1;
    }
    return res;
}

int main()
{

    return 0;
}
