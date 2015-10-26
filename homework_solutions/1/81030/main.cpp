#include <stdio.h>

float derCorput(unsigned n)
{
    float s = 0;
    unsigned c = 1;

    while (n)
    {
        c <<= 1;

        if (n & 1)
            s += 1.0f / c;

        n >>= 1;
    }

    return s;
}

int main()
{
    unsigned i;

    for (i = 0; i < 20; ++i)
        printf("%f, ", derCorput(i));

    return 0;
}
