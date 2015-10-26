#include <cstdio>

using namespace std;

float derCorput(int seed) {
    float r = 0;
    float base_inv = 0.5;

    while (seed != 0) {
        r = r + static_cast<float>(seed & 0x1) * base_inv;
        base_inv = base_inv / 2.f;
        seed >>= 1;
    }

    return r;
}


int main() {
    for (auto i = 0; i < 2000; ++i) {
        printf("%f, ", derCorput(i));
    }
    return 0;
}