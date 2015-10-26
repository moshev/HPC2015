#include <iostream>
#include <iomanip>

#include <chrono>

#include <cstdint>
#include <limits>

#include <cstring>

const uint64_t EXPONENT_BIAS = 1023;
const uint64_t MANTISSA_BIT_SIZE = 52;
const uint64_t ONE = 1;

const int MEMBERS_COUNT = 1000000;
const unsigned int BEGIN_INDEX = 0;
const int SHOW_MEMBERS = 20;

double derCorput(unsigned int number);
double derCorput2(unsigned int number);

int main()
{
    double* results = new double[MEMBERS_COUNT];
    double* results2 = new double[MEMBERS_COUNT];

    // init memory here, so the time for calculations is correct
    memset(results, 5, MEMBERS_COUNT * sizeof(double));
    memset(results2, 5, MEMBERS_COUNT * sizeof(double));

    auto beginTime = std::chrono::high_resolution_clock::now();

    for (unsigned int i = BEGIN_INDEX; i < BEGIN_INDEX + MEMBERS_COUNT; ++i)
    {
        results[i - BEGIN_INDEX] = derCorput(i);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto timeNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-beginTime).count();

    auto beginTime2 = std::chrono::high_resolution_clock::now();

    for (unsigned int i = BEGIN_INDEX; i < BEGIN_INDEX + MEMBERS_COUNT; ++i)
    {
        results2[i - BEGIN_INDEX] = derCorput2(i);
    }

    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto timeNanos2 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime2-beginTime2).count();

    std::cout << std::endl;
    std::cout << "Calculated " << MEMBERS_COUNT
        << " consecutive members of the Van der Corput sequence," << std::endl;
    std::cout << "starting with " << BEGIN_INDEX << "-th member." << std::endl << std::endl;
    std::cout << std::left << std::setw(25) << "Time, bitwise simple: " << std::right <<
        std::setw(20) << timeNanos << " ns." << std::endl << std::endl;
    std::cout << std::left << std::setw(25) << "Time, bitwise optimized: " << std::right <<
        std::setw(20) << timeNanos2 << " ns." << std::endl << std::endl;

    bool validationPassed = true;
    for (unsigned int i = 0; i < MEMBERS_COUNT; ++i)
    {
        if(results[i] != results2[i])
        {
            std::cout << "ERROR! The 2 algorithms returned different results for index: " <<
                i + BEGIN_INDEX << ". " << std::endl << "    algorithm1: " << results[i] <<
                 ", algorithm2: " << results2[i] << std::endl;
            validationPassed = false;
        }
    }
    if (validationPassed)
    {
        std::cout << "Validation passed! The 2 algorithms returned one and the same results." << std::endl;
    }

    std::cout << std::endl << "The first " << std::min(SHOW_MEMBERS, MEMBERS_COUNT) << " members:" << std::endl << std::endl;
    for (unsigned int i = 0; i < SHOW_MEMBERS && i < MEMBERS_COUNT; ++i)
    {
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 2)
            << results[i] << ", " << results2[i] << std::endl;
    }

    delete[] results;
    delete[] results2;

    return 0;
}

double derCorput(unsigned int number)
{
    if (number == 0) {
        return 0.0;
    }

    uint64_t bitwise_double = 0;
    uint64_t remainder = 0;

    uint64_t bits_all = 0;
    uint64_t bits_since_first_set_one = 0;

    while (number)
    {
        remainder = number & ONE;
        number >>= ONE;

        ++bits_all;

        if (bits_since_first_set_one)
        {
            bitwise_double |= (remainder << (MANTISSA_BIT_SIZE - bits_since_first_set_one));
            ++bits_since_first_set_one;
        }

        if (remainder && !bits_since_first_set_one)
        {
            bits_since_first_set_one = 1;
        }
    }

    uint64_t first_set_bit_position = bits_all - bits_since_first_set_one;
    uint64_t exponent = EXPONENT_BIAS - first_set_bit_position - 1;

    bitwise_double |= (exponent << (MANTISSA_BIT_SIZE));

    return *(reinterpret_cast<double*>(&bitwise_double));
}

static const unsigned char BitReverseTable256[256] =
{
#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
    R6(0), R6(2), R6(1), R6(3)
};

static const int MultiplyDeBruijnBitPosition[32] =
{
  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

inline unsigned int reverseBits(unsigned int straight)
{
    return  (BitReverseTable256[straight & 0xff] << 24) |
        (BitReverseTable256[(straight >> 8) & 0xff] << 16) |
        (BitReverseTable256[(straight >> 16) & 0xff] << 8) |
        (BitReverseTable256[(straight >> 24) & 0xff]);
}

inline unsigned int lastSetBitPosition(unsigned int number)
{
    return MultiplyDeBruijnBitPosition[((uint32_t)((number & -number) * 0x077CB531U)) >> 27];
}

double derCorput2(unsigned int number)
{
    if (number == 0) {
        return 0.0;
    }

    uint64_t bitwise_double = 0;

    // First after reversing, it is last before reversing
    uint64_t first_set_bit_position = lastSetBitPosition(number);

    uint64_t reversed_bits = reverseBits(number);
    // Ignore the first set bit, 1.f is assumed in the IEEE 754
    reversed_bits = reversed_bits & (~(ONE << (32 - first_set_bit_position - 1)));
    reversed_bits <<= (first_set_bit_position + 1);

    bitwise_double |= (reversed_bits << (MANTISSA_BIT_SIZE - 32));

    uint64_t exponent = EXPONENT_BIAS - first_set_bit_position - 1;
    bitwise_double |= (exponent << (MANTISSA_BIT_SIZE));

    return *(reinterpret_cast<double*>(&bitwise_double));
}

