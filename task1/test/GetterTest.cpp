#include "gtest/gtest.h"
#include "CircularBuffer.h"
#include "TestUtils.h"

TEST(GetterTest, SquareBrackets)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a[400], 321);
}

TEST(GetterTest, ConstSquareBrackets)
{
    const CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a[400], 321);
}

TEST(GetterTest, At)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a[10], 321);
}

TEST(GetterTest, AtException)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_ANY_THROW(a.at(-10));
}

TEST(GetterTest, ConstAt)
{
    const CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a.at(10), 321);
}

TEST(GetterTest, ConstAtException)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_ANY_THROW(a.at(-10));
}

TEST(GetterTest, GetFront)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a.front(), 321);
}

TEST(GetterTest, GetConstFront)
{
    const CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a.front(), 321);
}

TEST(GetterTest, GetBack)
{
    CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a.back(), 321);
}

TEST(GetterTest, GetConstBack)
{
    const CircularBuffer<int> a(500, 321);
    EXPECT_EQ(a.back(), 321);
}

TEST(GetterTest, GetSize)
{
    CircularBuffer<int> a(5);
    for (int i = 0; i < 3; i++)
        a.push_back(i+1);
    EXPECT_EQ(a.size(), 3);
}

TEST(GetterTest, Empty)
{
    CircularBuffer<int> a(5);
    for (int i = 0; i < 3; i++)
        a.push_back(i+1);
    for (int i = 0; i < 3; i++)
        a.pop_back();
    EXPECT_TRUE(a.empty());
}

TEST(GetterTest, Full)
{
    CircularBuffer<int> a(5);
    for (int i = 0; i < 5; i++)
        a.push_back(i+1);
    EXPECT_TRUE(a.full());
}

TEST(GetterTest, Reserve)
{
    CircularBuffer<int> a(5);
    for (int i = 0; i < 3; i++)
        a.push_back(i+1);
    EXPECT_EQ(a.reserve(), 2);
}

TEST(GetterTest, Capacity)
{
    CircularBuffer<int> a(5);
    EXPECT_EQ(a.capacity(), 5);
}
