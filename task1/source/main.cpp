#include "CircularBuffer.h"

int main()
{
    CircularBuffer<int> a(100);
    CircularBuffer<int> b(100);
    for (int i = 0; i < 100; i++)
    {
        a[i] = i;
        b[i] = i + 1;
        std::cout << a[i] << " " << b[i] << std::endl;
    }
    std::cout << (a == b);
}