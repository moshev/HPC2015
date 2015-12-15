#include <stdint.h>
#include <string.h>

#ifdef _MSC_VER
#define BYTESWAP64 _byteswap_uint64
#else
#define BYTESWAP64 __builtin_bswap64
#endif
#define BUF_SIZE 8

void reverse(char *bytes, int numChunks) {
  uint64_t *arr = (uint64_t *)bytes;
  uint64_t buf[BUF_SIZE];
  for (int j = 0; j < numChunks; ++j) {
    memcpy(buf, arr + BUF_SIZE * j, BUF_SIZE * sizeof(uint64_t));
    for (int i = 0; i < BUF_SIZE; ++i)
      buf[i] = BYTESWAP64(buf[i]);
    for (int i = 0; i < BUF_SIZE / 2; ++i) {
      uint64_t tmp = buf[i];
      buf[i] = buf[BUF_SIZE - i - 1];
      buf[BUF_SIZE - i - 1] = tmp;
    }
    memcpy(arr + BUF_SIZE * j, buf, BUF_SIZE * sizeof(uint64_t));
  }
}
int main() { return 0; }
