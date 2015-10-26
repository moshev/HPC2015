//Homework 1 - by Nikolay Genov 44909
#define DOUBLE_MANTISSA_SIZE 52
#define DOUBLE_EXP_SIZE 11
#define EXP_START 1022
#define UINT_BIT_SIZE 32

struct real {
  unsigned long long man : DOUBLE_MANTISSA_SIZE;
  unsigned exp : DOUBLE_EXP_SIZE;
  unsigned sign : 1;
};

union S {
  double f;
  real r;
};

unsigned int bitReverseWithLookUp(unsigned int n) {
  static const unsigned char BitReverseTable256[256] = {
#define R2(n) n, n + 2 * 64, n + 1 * 64, n + 3 * 64
#define R4(n) R2(n), R2(n + 2 * 16), R2(n + 1 * 16), R2(n + 3 * 16)
#define R6(n) R4(n), R4(n + 2 * 4), R4(n + 1 * 4), R4(n + 3 * 4)
      R6(0), R6(2), R6(1), R6(3)};

  return (BitReverseTable256[n & 0xff] << 24) | (BitReverseTable256[(n >> 8) & 0xff] << 16) |
         (BitReverseTable256[(n >> 16) & 0xff] << 8) | (BitReverseTable256[(n >> 24) & 0xff]);
}

int countConsecutiveTrailingZeroBits(unsigned int n) {
  static const int MultiplyDeBruijnBitPosition[32] = {0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
                                                      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};
  return MultiplyDeBruijnBitPosition[((unsigned int)((n & -n) * 0x077CB531U)) >> 27];
}

float derCorput(unsigned n) {
  if (!n)
    return 0;
  unsigned reversed = bitReverseWithLookUp(n);
  int numberOfTrailingZeroes = countConsecutiveTrailingZeroBits(n);
  int numberOfTrailingZeroesInReversed = countConsecutiveTrailingZeroBits(reversed);
  unsigned numberOfBitsInOriginalNumber = UINT_BIT_SIZE - numberOfTrailingZeroesInReversed;
  reversed >>= numberOfTrailingZeroesInReversed;
  unsigned numberOfBitsInReversedNumber = numberOfBitsInOriginalNumber - numberOfTrailingZeroes - 1;
  unsigned long long mantissaShiftNumber = reversed ^ (1 << numberOfBitsInReversedNumber);

  S s = {0};
  s.r.man = mantissaShiftNumber << (DOUBLE_MANTISSA_SIZE - numberOfBitsInReversedNumber);
  s.r.exp = EXP_START - numberOfTrailingZeroes;
  return s.f;
}

int main() { return 0; }

