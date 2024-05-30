#include <vector>
#include "CircularBuffer.h"

template<typename T>
bool isBufEqVector(const CircularBuffer<T> &Buf, const std::vector<T> &Vec)
{
    if (Buf.size() != Vec.size())
        return false;

    for (int i = 0; i < Buf.size(); i++)
    {
        if (Buf[i] != Vec[i])
            return false;
    }
    return true;
}

