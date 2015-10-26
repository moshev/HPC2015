#include <cmath>
#include <cstdio>

double derCorput(unsigned n)
{
    double derCorput = 0, denominator = 1;
    while (n)
    {
        derCorput += (n%2) / (denominator*=2);
        n/=2;
    }
    return derCorput;
}


int main()
{

    for (int i = 0; i < 20; ++i)
   printf("%f, ", derCorput(i));
    return 0;
}
