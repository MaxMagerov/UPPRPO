#include "gtest/gtest.h"
#include "CircularBuffer.h"
#include "TestUtils.h"

TEST(LinearizeTest, TrivialLinearize) {
    CircularBuffer<int> buf(1000, 123);
    buf[0] = 500;
    EXPECT_EQ(*buf.linearize(), 500);
}

TEST(LinearizeTest, OrderedLinearize) {
    CircularBuffer<int> buf(5);
    buf.push_back(1);
    buf.push_back(2);
    buf.push_back(3);
    buf.push_back(4);
    buf.pop_front();
    EXPECT_TRUE(isBufEqVector(buf, std::vector<int>{2,3,4}));
    EXPECT_EQ(buf.linearize()[0], 2);
}

TEST(LinearizeTest, NotOrderedLinearize) {
    CircularBuffer<int> buf(5, 123);
    buf.push_back(1);
    buf.push_back(2);
    buf.push_back(3);
    buf.push_front(4);
    EXPECT_TRUE(isBufEqVector(buf, std::vector<int>{4, 123, 123, 1, 2}));
    EXPECT_EQ(buf.linearize()[0], 4);
}

TEST(LinearizeTest, IsLinearized){
    CircularBuffer<int> buf(3);
    buf.push_back(1);
    buf.push_back(2);
    buf.push_back(3);
    EXPECT_TRUE(buf.is_linearized());
    buf.push_back(4);
    EXPECT_FALSE(buf.is_linearized());
}

