#include <iostream>

/*
    Martin Stoev
    FN: 855240
*/
float derCorput(unsigned number)
{
    unsigned mask = 1;
    unsigned long power = 2;
    float output = 0;

    while(mask <= number)
    {
        if(number & mask)
            output += 1./(float)(power);
        power <<= 1;
        mask <<= 1 ;
    }
    return output;
}
int main()
{
    for(int i = 0 ; i < 20 ; i++)
        std::cout << derCorput(i) << ' ';
    return 0;
}
