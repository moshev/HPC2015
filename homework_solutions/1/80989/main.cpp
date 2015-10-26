#include <cstdio>
#include <iostream>

using namespace std;

float derCorput(unsigned n)
{
    if(n == 0)
        return 0;
    //float res = 0.1;

    unsigned temp = 0;
    short ctn = 0, ctn2 = 0;
    unsigned n_copy = n;

    while(n_copy != 0)
    {
        n_copy /= 2;
        ++ctn;
    }

    unsigned to_be_shifted = 1;

    while(n != 0)
    {
        short ctn0 = ctn2;
        unsigned n0 = n;

        if(n0%2 == 1)
            temp += ( to_be_shifted << ctn - ctn2 - 1);
        n /= 2;
        ++ctn2;
    }

    return (float)temp/(to_be_shifted << ctn);
}

int main()
{
    for (auto i = 0; i < 20; ++i)
    printf("%f, ", derCorput(i));
    return 0;
}
