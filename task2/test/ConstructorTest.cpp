#include "gtest/gtest.h"
#include "CircularBuffer.h"
#include "TestUtils.h"

TEST(ConstructorTest, DefaultConstructor) {
    EXPECT_NO_THROW({
                        CircularBuffer<int> buf;
                    });
}

TEST(ConstructorTest, ConstructorWithCapacity) {
    CircularBuffer<int> buf(1000);
    EXPECT_EQ(buf.capacity(), 1000);
}

TEST(ConstructorTest, ConstructorWithCapacityElem) {
    CircularBuffer<int> buf(1000, 123);
    EXPECT_EQ(buf.capacity(), 1000);
    std::vector<int> vec(1000, 123);
    EXPECT_TRUE(isBufEqVector(buf, vec));
}

TEST(ConstructorTest, CopyConstructor) {
    CircularBuffer<int> buf1(1000, 123);
    CircularBuffer<int> buf2(buf1);
    EXPECT_EQ(buf1.size(), buf2.size());
    EXPECT_EQ(buf1.capacity(), buf2.capacity());
    EXPECT_EQ(buf1[0], buf2[0]);
    EXPECT_EQ(buf1[10], buf2[10]);
}
