//    Мартин Стоев
//    Информационни системи
//    855240
//
//    HPC-Домашно 3
#include <iostream>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <immintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>

#define LARGE_INTEGER unsigned long long
#define BOOL bool

inline void reverse(char* bytes, int numChunks)
{
    static __m256i reversed1;
    static __m256i reversed2;

    for(size_t i = 0; i < numChunks; ++i)
    {
        reversed1 = _mm256_set_epi8(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
                                    bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15], bytes[16],
                                    bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23], bytes[24],
                                    bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31]);

		reversed2 = _mm256_set_epi8(bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
                                    bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
                                    bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
                                    bytes[56], bytes[57], bytes[58], bytes[59], bytes[60], bytes[61], bytes[62], bytes[63]);


        _mm256_storeu_si256((__m256i*)&bytes[0], reversed2);
        _mm256_storeu_si256((__m256i*)&bytes[32], reversed1);
        bytes += 64;
    }
}
#include <xmmintrin.h>
void takeTime(void (pn)(char* bytes, int numChunks), char* bytes, int numChunks, size_t cycles)
{
    clock_t start = clock();

    for(size_t i = 0 ; i < cycles ; ++i)
    {
        pn(bytes, numChunks);
    }
    clock_t end = clock();

    std::cout << (end - start) << std::endl;
}

static uint64_t timer_nsec() {

		return 1000000000;
}

static void foo(
	float (& inout)[8]) {

	static const size_t idx[][2] = {
		{ 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 },
		{ 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 },
		{ 1, 2 }, { 5, 6 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
		{ 2, 4 }, { 3, 5 },
		{ 1, 2 }, { 3, 4 }, { 5, 6 }
	};
    static const size_t length = sizeof(idx) / sizeof(idx[0]);
	for (size_t i = 0; i < length; ++i) {
        if(inout[idx[i][0]] > inout[idx[i][1]])
        {
            const float temp = inout[idx[i][0]];
            inout[idx[i][0]] = inout[idx[i][1]];
            inout[idx[i][1]] = temp;
        }
	}
}

static void bar(float (& inout)[8]) {
    static __m128 first;
    static __m128 second;
    static __m128 cmp1;
    static __m128 cmp2;
    static __m128 res1;
    static __m128 res2;
    static __m128 temp;
    static __m128 res3;
    static __m128 res4;
    static float result1[4];
    static float result2[4];

	const size_t idx[][2] = {
		{0, 1}, {3, 2}, {4, 5}, {7, 6},
		{0, 2}, {1, 3}, {6, 4}, {7, 5},
		{0, 1}, {2, 3}, {5, 4}, {7, 6},
		{0, 4}, {1, 5}, {2, 6}, {3, 7},
		{0, 2}, {1, 3}, {4, 6}, {5, 7},
		{0, 1}, {2, 3}, {4, 5}, {6, 7}
	};
    // 24 = sizeof(idx)/sizeof(idx[0])
    for(int i = 0 ; i < 24 ; i+=4)
    {   // the first and second are packed vectors of the i-th element to the i-th +3
        // reversed because the _mm_set_ps() reverses the data for some reasons
        first = _mm_set_ps(inout[idx[i+3][0]], inout[idx[i+2][0]], inout[idx[i+1][0]], inout[idx[i][0]]);
        second = _mm_set_ps(inout[idx[i+3][1]], inout[idx[i+2][1]], inout[idx[i+1][1]], inout[idx[i][1]]);

        // cmpge because if cmpgt(greater then) it will be bugged for array with equal data insside ex [1,1,1,1,1] -> [0,0,0,0]
        cmp1 = _mm_cmpge_ps(first, second);
        cmp2 = _mm_cmpge_ps(second, first);

        // the formula
        // x = (c & y) | (!c & x)
        // y = (c & x) | (!c & y)
        // where x and y are elements
        res1 = _mm_and_ps(second, cmp1);
        res2 = _mm_and_ps(first, cmp2);

        res3 = _mm_and_ps(first, cmp1);
        res4 = _mm_and_ps(second, cmp2);

        first = _mm_or_ps(res1, res2);
        second = _mm_or_ps(res3, res4);

        // put them on the positions
        _mm_storeu_ps(result1, first);
        _mm_storeu_ps(result2, second);

        for(int j = 0 ; j < 4 ; ++j)
        {
            inout[idx[i+j][0]] = result1[j];
            inout[idx[i+j][1]] = result2[j];
        }
    }
}

static void insertion_sort(
	float (& inout)[8]) {

	for (size_t i = 1; i < 8; ++i) {
		size_t pos = i;
		const float val = inout[pos];

		while (pos > 0 && val < inout[pos - 1]) {
			inout[pos] = inout[pos - 1];
			--pos;
		}
		inout[pos] = val;
	}
}


static size_t verify(
	const size_t count,
	float* const input) {

	assert(0 == count % 8);

	for (size_t i = 0; i < count; i += 8)
		for (size_t j = 0; j < 7; ++j)
			if (input[i + j] > input[i + j + 1])
				return i + j;

	return -1;
}

int main()
{
    const size_t n = 128;
    char bytes[n];
    for (int i = 0; i < n; ++i)
        bytes[i] = i;
    //for(int i = 0 ; i < 1 ; ++i)
    reverse(bytes, (n/64));

    float arr[8];
    for(int i = 0 ; i < 8; ++i)
        arr[i] = rand() % 50;

    bar(arr);

    for(int i = 0 ; i < 8; ++i)
        std::cout << arr[i] << " ";

    return 0;
}

//int main(
//	int argc,
//	char** argv) {
//
//	unsigned alt = 1;
//	const bool err = argc > 2 || argc == 2 && 1 != sscanf(argv[1], "%u", &alt);
//
//	if (err || alt > 2) {
//		std::cerr << "usage: " << argv[0] << " [opt]\n"
//			"\t0 foo (default)\n"
//			"\t1 bar\n"
//			"\t2 insertion_sort\n"
//			<< std::endl;
//		return -3;
//	}
//
//	const size_t count = 1 << 25;
//	float* const input = (float*) malloc(sizeof(float) * count + 63);
//	float* const input_aligned = reinterpret_cast< float* >(uintptr_t(input) + 63 & -64);
//
//	std::cout << std::hex << std::setw(8) << input << " (" << input_aligned << ") : " << std::dec << count << " elements" << std::endl;
//
//	for (size_t i = 0; i < count; ++i)
//		input_aligned[i] = rand() % 42;
//
//	uint64_t t0;
//	uint64_t t1;
//    std::cout << alt<< std::endl;
//	switch (alt) {
//	case 0: // foo
//		{
//			t0 = timer_nsec();
//
//			for (size_t i = 0; i < count; i += 8)
//				foo(*reinterpret_cast< float (*)[8] >(input_aligned + i));
//
//			t1 = timer_nsec();
//
//			const size_t err = verify(count, input_aligned);
//			if (-1 != err)
//				std::cerr << "error at " << err << std::endl;
//		}
//		break;
//
//	case 1: // bar
//		{
//			t0 = timer_nsec();
//
//			for (size_t i = 0; i < count; i += 8)
//				bar2(*reinterpret_cast< float (*)[8] >(input_aligned + i));
//
//			t1 = timer_nsec();
//
//			const size_t err = verify(count, input_aligned);
//			if (-1 != err)
//				std::cerr << "error at " << err << std::endl;
//		}
//		break;
//
//	case 2: // insertion_sort
//		{
//			t0 = timer_nsec();
//
//			for (size_t i = 0; i < count; i += 8)
//				insertion_sort(*reinterpret_cast< float (*)[8] >(input_aligned + i));
//
//			t1 = timer_nsec();
//
//			const size_t err = verify(count, input_aligned);
//			if (-1 != err)
//				std::cerr << "error at " << err << std::endl;
//		}
//		break;
//	}
//
//	const double sec = double(t1 - t0) * 1e-9;
//	std::cout << "elapsed time: " << sec << " s" << std::endl;
//
//	free(input);
//	return 0;
//}

