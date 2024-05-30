#include "gtest/gtest.h"
#include "CircularBuffer.h"
#include "TestUtils.h"

TEST(GetterTest, TrivialEq)
{
    CircularBuffer<int> a(5,12);
    EXPECT_TRUE(a == a);
}

TEST(GetterTest, DifSizeEq)
{
    CircularBuffer<int> a(5);
    a.push_back(1);
    a.push_back(2);
    CircularBuffer<int> b(5);
    b.push_back(1);
    b.push_back(2);
    b.push_back(3);
    b.push_back(4);
    EXPECT_FALSE(a == b);
}

TEST(GetterTest, DifCapacityEq)
{
    CircularBuffer<int> a(5,3);
    CircularBuffer<int> b(7,3);
    EXPECT_FALSE(a == b);
}

TEST(GetterTest, EqualityFalse)
{
    CircularBuffer<int> a(100);
    CircularBuffer<int> b(100);
    for (int i = 0; i < 100; i++)
    {
        a.push_back(i);
        b.push_back(i);
    }
    b[96] = 502;
    EXPECT_FALSE(a == b);
}

TEST(GetterTest, TrivialUneq)
{
    CircularBuffer<int> a(5,12);
    EXPECT_FALSE(a != a);
}

TEST(GetterTest, DifSizeUneq)
{
    CircularBuffer<int> a(5);
    a.push_back(1);
    a.push_back(2);
    CircularBuffer<int> b(5);
    b.push_back(1);
    b.push_back(2);
    b.push_back(3);
    b.push_back(4);
    EXPECT_TRUE(a != b);
}

TEST(GetterTest, DifCapacityUneq)
{
    CircularBuffer<int> a(5,3);
    CircularBuffer<int> b(7,3);
    EXPECT_TRUE(a != b);
}

TEST(GetterTest, UnequalityFalse)
{
    CircularBuffer<int> a(100);
    CircularBuffer<int> b(100);
    for (int i = 0; i < 100; i++)
    {
        a.push_back(i);
        b.push_back(i);
    }
    b[96] = 502;
    EXPECT_TRUE(a != b);
}