#if defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
    #define M_ALWAYS_INLINE __inline__ __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define M_ALWAYS_INLINE __forceinline
#else
    #error "Compiler not supported."
#endif // Специфични за компилатора неща

#include <cstdio>
#include <utility>
#include <cstddef>


// Създава група от индекси с която ще може да инициализираме
// масив по време на компилация, чрез constexpr функция

template <size_t... tail>
struct ct_Natural {};                   // базовата структура

template<size_t N, size_t... tail>
struct ct_Natural_gen :                 // генерира редицата индекси (  естествените числа от 1 до N)
ct_Natural_gen<N-1,N,tail...> {};

template<size_t... tail>
struct ct_Natural_gen<0,tail...> :      // дъно на рекурсията
ct_Natural<tail...> {};

// Сега вече ct_Natural_gen<N> има за предшественик ct_Natural<1,2,..,N-1,N>

template<typename T_array,size_t N>
struct ct_array_holder
{
    T_array data[N];
};

template<typename T_array,typename T_function, size_t... tail>
constexpr ct_array_holder<T_array,sizeof...(tail)>
ct_array_map(ct_Natural<tail...> a,T_array b,T_function f)
{
    return {{ f(tail)... }};
}

template<typename T_array, typename T_function,T_function f, size_t array_size>
struct ct_array
{
    static constexpr ct_array_holder<T_array,array_size> data = ct_array_map(ct_Natural_gen<array_size>(),T_array(),f);
};

template<typename T_array, typename T_function,T_function f, size_t array_size>
 constexpr ct_array_holder<T_array,array_size>
ct_array<T_array,T_function,f,array_size>::data;

// Сега идеята е да се направи масив в compile-time с
// предварително изчислени битови макси и 2^(-k-1) за
// пресмятане на число от редицата.

// Ще считаме че истинския индекс на цифрата е n-1, тъй като индексацията
// в масива започва от 1, а не от 0.

constexpr std::pair<float,unsigned> der_Corput_array_map(size_t n)
    {
        return (n==1) ? std::make_pair(1.0f/2.0f,size_t(1)) :
                         std::make_pair(der_Corput_array_map(n-1).first/2.0f,size_t(1)<<(size_t(n-1)));
    }
    
typedef ct_array<std::pair<float,unsigned>,decltype(&der_Corput_array_map),der_Corput_array_map,sizeof(unsigned)*8> static_data;
// Сега вече имаме статичните данни.

template<size_t iteration>
struct derCorput_unroller
{
    
static M_ALWAYS_INLINE void run(unsigned n, float& result)
{
    result+=static_data::data.data[iteration].first*float((n & static_data::data.data[iteration].second)!=0);
    derCorput_unroller<iteration-1>::run(n,result);
}

};

template<>
struct derCorput_unroller<0>
{

static M_ALWAYS_INLINE void run(unsigned n, float& result)
{
    result+=static_data::data.data[0].first*float((n & static_data::data.data[0].second)!=0);
}

};

M_ALWAYS_INLINE float derCorput(unsigned n)
{
    float result=0.0f;
    
    derCorput_unroller<sizeof(unsigned)*8-1>::run(n,result);
    
    return result;
}


int main()
{
    for (auto i = 0; i < 20; ++i)
        printf("%f , ",derCorput(i));
}
