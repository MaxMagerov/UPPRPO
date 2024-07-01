#include "gtest/gtest.h"
#include "CircularBuffer.h"
#include "TestUtils.h"

TEST(ChangeTest, SetZeroCapacity)
{
CircularBuffer<int> a(5, 123);
a.set_capacity(0);
EXPECT_EQ(a.capacity(), 0);
}

TEST(ChangeTest, SetSameCapacity)
{
CircularBuffer<int> a(5, 123);
a.set_capacity(5);
EXPECT_EQ(a.capacity(), 5);
}

TEST(ChangeTest, SetGreaterCapacity) {
    CircularBuffer<int> a(5, 123);
    a.set_capacity(15);
    EXPECT_EQ(a.capacity(), 15);
}

TEST(ChangeTest, SetLesserCapacity) {
    CircularBuffer<int> a(5, 123);
    a.set_capacity(3);
    EXPECT_EQ(a.capacity(), 3);
}

TEST(ChangeTest, SetSizeGreaterThanCapacity) {
    EXPECT_ANY_THROW(
            {
                CircularBuffer<int> a(5, 123);
                a.resize(10);
            }
    );
}

TEST(ChangeTest, SetSameSize)
{
    CircularBuffer<int> a(5, 123);
    a.resize(5);
    EXPECT_EQ(a.size(), 5);
}

TEST(ChangeTest, SetGreaterSize)
{
    CircularBuffer<int> a(5);
    a.push_back(1);
    a.push_back(2);
    a.resize(4);
    EXPECT_EQ(a.size(), 4);
}

TEST(ChangeTest, SetLesserSize)
{
    CircularBuffer<int> a(5, 123);
    a.resize(3);
    EXPECT_EQ(a.size(), 3);
}

TEST(ChangeTest, TrivialAssignment)
{
    EXPECT_NO_THROW({
        CircularBuffer<int> a(5, 123);
        a = a;
    });
}

TEST(ChangeTest, Assignment)
{
    CircularBuffer<int> a(5, 123);
    CircularBuffer<int> b(7, 321);
    a.operator=(b);
    EXPECT_EQ(a.capacity(), 7);
}

TEST(ChangeTest, Swap)
{
    EXPECT_NO_THROW({
                        CircularBuffer<int> a(5, 123);
                        CircularBuffer<int> b(7, 321);
                        a.swap(b);
                    });
}

TEST(ChangeTest, ZeroCapacityPushBack)
{
    EXPECT_ANY_THROW({
        CircularBuffer<int> a;
        a.push_back(5);
    });
}

TEST(ChangeTest, ZeroCapacityPushFront)
{
    EXPECT_ANY_THROW({
                         CircularBuffer<int> a;
                         a.push_front(5);
                     });
}

TEST(ChangeTest, SizeEqCapacityPushFront)
{
    CircularBuffer<int> a(3,1);
    a.push_front(2);
    EXPECT_TRUE(isBufEqVector(a,std::vector<int> {2,1,1}));
}

TEST(ChangeTest, SizeNotEqCapacityPushFront)
{
    CircularBuffer<int> a(5);
    a.push_back(1);
    a.push_back(2);
    a.push_front(3);
    EXPECT_TRUE(isBufEqVector(a,std::vector<int> {3,1,2}));
}

TEST(ChangeTest, ZeroCapInsert)
{
    EXPECT_ANY_THROW({
        CircularBuffer<int> a;
        a.insert(0);
    });
}

TEST(ChangeTest, FullBufInsert)
{
    EXPECT_ANY_THROW({
                         CircularBuffer<int> a(5,1);
                         a.insert(2);
                     });
}

TEST(ChangeTest, InsertOut)
{
    EXPECT_ANY_THROW({
                         CircularBuffer<int> a(5);
                         a.push_back(1);
                         a.push_back(2);
                         a.insert(4,3);
                     });
}

TEST(ChangeTest, Insert)
{
    CircularBuffer<int> a(6);
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    a.push_back(4);
    a.insert(2, 5);
    EXPECT_TRUE(isBufEqVector(a,std::vector<int> {1,2,5,3,4}));
}

TEST(ChangeTest, Erase)
{
    CircularBuffer<int> a(10);
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    a.push_back(4);
    a.push_back(5);
    a.push_back(6);
    a.push_back(7);
    a.push_back(8);
    a.erase(2,5);
    EXPECT_TRUE(isBufEqVector(a,std::vector<int> {1,2,6,7,8}));
}

TEST(ChangeTest, TrivialClear)
{
    EXPECT_NO_THROW({
       CircularBuffer<int> a;
       a.clear();
    });
}

TEST(ChangeTest, Clear)
{
    EXPECT_NO_THROW({
                        CircularBuffer<int> a(1000,123);
                        a.clear();
                    });
}