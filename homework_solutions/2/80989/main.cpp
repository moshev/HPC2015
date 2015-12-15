#include <cstdio>

void reverse(char* bytes, int numChunks)
{
    for(int i = 0; i < numChunks; ++i)
    {
        for(int j = 0; j < 32 ; ++j)
        {
            if((bytes + i*64)[j] != (bytes + i*64)[63-j])
            {
                (bytes + i*64)[j] = (bytes + i*64)[j]^(bytes + i*64)[63-j];
                (bytes + i*64)[63-j] = (bytes + i*64)[j]^(bytes + i*64)[63-j];
                (bytes + i*64)[j] = (bytes + i*64)[j]^(bytes + i*64)[63-j];
            }
        }
    }
}

int main()
{
    char bytes[128];
    for (int i = 0; i < 128; ++i)
        bytes[i] = i;

    reverse(bytes, (128/64));

    printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]);
    return 0;
}
