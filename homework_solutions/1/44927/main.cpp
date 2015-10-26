/*
 * Имплементирайте функцията float derCorput(unsigned n);
 * която връща n-тото число от редицата на Дер Корпут, при основа 2.
 */

#include <iostream>
#include <cstdint>


unsigned long long fastPow(unsigned long long base, size_t degree)
{
    if(degree == 0)
    {
        return 1;
    }

    return (degree % 2 == 0) ? fastPow(base * base, degree / 2) : base * fastPow(base, degree - 1);
}

const uint8_t MAX_BITS = 32;
const uint8_t BASE = 2;
float MEMOIZE[MAX_BITS] = {0, };


float derCorput(unsigned nth)
{
    float sum = 0;

    for(auto index = 0; index < MAX_BITS; ++index)
    {
       if(nth & (1 << index))
       {
            if(MEMOIZE[index] == 0)
            {
                MEMOIZE[index] = static_cast<float>(1) / (1 << index + 1);//(float)fastPow(BASE, index + 1);
            }
            sum += MEMOIZE[index];
       }
    }
   return sum;
}

/*
int main()
{
    for(int i = 0; i < 100; ++i)
    {
        std::cout << derCorput(i) << std::endl;
    }

    return 0;
}

*/


