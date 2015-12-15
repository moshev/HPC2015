#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cassert>

// AVX за втората задача
#include <immintrin.h>

// SSE2 и SSSE3 за първата
#include <emmintrin.h>
#include <tmmintrin.h>

#include <xmmintrin.h>

#if defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
    #define M_ALWAYS_INLINE __inline__ __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define M_ALWAYS_INLINE __forceinline
#else
    #error "Compiler not supported."
#endif // Специфични за компилатора неща


M_ALWAYS_INLINE
static void bar(float (& input)[8]) 
{
    
    /*
	static constexpr uint_fast8_t idx[][2] = {
		{0, 1}, {3, 2}, {4, 5}, {7, 6}, // (1)
		{0, 2}, {1, 3}, {6, 4}, {7, 5}, // (2)
		{0, 1}, {2, 3}, {5, 4}, {7, 6}, // (3)
		{0, 4}, {1, 5}, {2, 6}, {3, 7}, // (4)
		{0, 2}, {1, 3}, {4, 6}, {5, 7}, // (5)
		{0, 1}, {2, 3}, {4, 5}, {6, 7} // (6)
	};
    */
    // Индекса трябва да представим в по удобен вид за
    // AVX инструкциите. Няма смисъл от цикъл и после развиване
    // защото (4)-тия случай е специален... По добре на ръка.
    
    static constexpr int blend_mask_1 =0b10011001;
    static constexpr int blend_mask_2=0b11000011;
    static constexpr int blend_mask_3 =0b10100101;
    static constexpr int blend_mask_4 =0b00001111;
    static constexpr int blend_mask_5=0b00110011;
    static constexpr int blend_mask_6=0b01010101;
    
    // Отговаря на (1), (3) и (6)
    static constexpr int permute_mask_1=0b10110001;
    
    
    // Отговаря на (2) и (5)
    static constexpr int permute_mask_2=0b01001110;
    
    
    __m256 result= _mm256_load_ps(input);
    
    // (1)  
    
    __m256 mapped=_mm256_permute_ps(result,permute_mask_1);
    
    __m256 min=_mm256_min_ps(result,mapped);
    __m256 max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_1);
    
    // (2)
    
    mapped=_mm256_permute_ps(result,permute_mask_2);
    
    min=_mm256_min_ps(result,mapped);
    max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_2);
    
    // (3)
    
    mapped=_mm256_permute_ps(result,permute_mask_1);
    
    min=_mm256_min_ps(result,mapped);
    max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_3);
    
    // (4) Специалния случай тук трябва да пермутираме
    // между двете половини на YMM регистъра.
    
    mapped=_mm256_permute2f128_ps(result,result,1);
   
    min=_mm256_min_ps(result,mapped);
    max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_4);
   
    // (5)
    
    mapped=_mm256_permute_ps(result,permute_mask_2);
    
    min=_mm256_min_ps(result,mapped);
    max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_5);
    
    // (6)
    
    mapped=_mm256_permute_ps(result,permute_mask_1);
    
    min=_mm256_min_ps(result,mapped);
    max=_mm256_max_ps(result,mapped);
    
    result=_mm256_blend_ps(max,min,blend_mask_6);
     /**/
    _mm256_store_ps(input,result);
}

M_ALWAYS_INLINE
__m128i reverse_128_bit(char * addr,__m128i byte_order_mask)
{
    __m128i data_128=_mm_load_si128(reinterpret_cast<__m128i*>(addr));
                                           
    return _mm_shuffle_epi8(data_128,byte_order_mask);
}


static void reverse(char* bytes, int numChunks)
{
    // Маска за разместване на байтовете вътре в 128 битово парче
    
    static constexpr int chunk_of_32_bits_0=0x0C0D0E0F;
    static constexpr int chunk_of_32_bits_1=0x08090A0B;
    static constexpr int chunk_of_32_bits_2=0x04050607;
    static constexpr int chunk_of_32_bits_3=0x00010203;
    
    
    __m128i byte_order_mask=_mm_set_epi32(chunk_of_32_bits_3,
                                           chunk_of_32_bits_2,
                                           chunk_of_32_bits_1,
                                           chunk_of_32_bits_0);
                                           
    static constexpr int chunk_size=64;
    static constexpr int quarter_chunk_size=chunk_size/4;
    
    int end=chunk_size*numChunks;
    for(int i=0;i<end;i+=chunk_size)
    {
        // Сега ще извършим разместване на байтовете във всяка
        // 128-битова половина като ги подреждаме правилно.
        static constexpr int i_first_chunk_in=3*quarter_chunk_size;
        static constexpr int i_second_chunk_in=2*quarter_chunk_size;
        static constexpr int i_third_chunk_in=quarter_chunk_size;
        
        __m128i fourth_chunk=reverse_128_bit(bytes+i,byte_order_mask);
        
        __m128i third_chunk=reverse_128_bit(bytes+i+i_third_chunk_in,byte_order_mask);
        
        __m128i second_chunk=reverse_128_bit(bytes+i+i_second_chunk_in,byte_order_mask);
        
        __m128i first_chunk=reverse_128_bit(bytes+i+i_first_chunk_in,byte_order_mask);

        
        
        // Сега записваме резултата
        _mm_store_si128(reinterpret_cast<__m128i*>(bytes+i),first_chunk);

        _mm_store_si128(reinterpret_cast<__m128i*>(bytes+i+i_third_chunk_in),second_chunk);

        _mm_store_si128(reinterpret_cast<__m128i*>(bytes+i+i_second_chunk_in),third_chunk);

        _mm_store_si128(reinterpret_cast<__m128i*>(bytes+i+i_first_chunk_in),fourth_chunk);
       
    }
}


static void foo(
	float (& inout)[8]) {

	const size_t idx[][2] = {
		{ 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 },
		{ 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 },
		{ 1, 2 }, { 5, 6 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
		{ 2, 4 }, { 3, 5 },
		{ 1, 2 }, { 3, 4 }, { 5, 6 }
	};

	for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); ++i) {
		const float x = inout[idx[i][0]];
		const float y = inout[idx[i][1]];

		inout[idx[i][0]] = std::min(x, y);
		inout[idx[i][1]] = std::max(x, y);
	}
}

