#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>

template<typename T>
class CircularBuffer {
    T *buffer_;
    int begin_;
    int end_;
    int size_;
    int capacity_;
public:
    CircularBuffer()
            : buffer_{},
              begin_{},
              end_{},
              size_{},
              capacity_{} {}

    ~CircularBuffer() {
        if (capacity_ == 0)
            return;

        delete[](buffer_);
    }

    CircularBuffer(const CircularBuffer &cb)
            : buffer_{new T[cb.capacity_]},
              begin_{},
              end_{},
              size_{},
              capacity_{cb.capacity_} {
        for (int i = 0; i < cb.size_; i++)
            push_back(cb[i]);
    };

    //Конструирует буфер заданной ёмкости.
    explicit CircularBuffer(int capacity) : CircularBuffer() {
        capacity_ = capacity;
        buffer_ = new T[capacity];
    }

    //Конструирует буфер заданной ёмкости, целиком заполняет его элементом elem.
    CircularBuffer(int capacity, const T &elem) : CircularBuffer(capacity) {
        size_ = capacity_;
        end_ = capacity - 1;
        for (int i = 0; i < capacity; i++)
            buffer_[i] = elem;
    }

    //Доступ по индексу. Не проверяют правильность индекса.
    T &operator[](int i) {
        return buffer_[(begin_ + i) % capacity_];
    }
    const T &operator[](int i) const {
        return buffer_[(begin_ + i) % capacity_];
    }

    //Доступ по индексу. Методы бросают исключение в случае неверного индекса.
    T &at(int i) {
        if (i < 0 || i >= size_) {
            throw std::out_of_range("Invalid index");
        }
        return buffer_[(begin_ + i) % capacity_];
    }
    const T &at(int i) const {
        if (i < 0 || i >= size_) {
            throw std::out_of_range("Invalid index");
        }
        return buffer_[(begin_ + i) % capacity_];
    }

    //Ссылка на первый элемент.
    T &front() {
        return buffer_[begin_];
    }
    //Ссылка на последний элемент.
    T &back() {
        return buffer_[end_];
    }
    const T &front() const {
        return buffer_[begin_];
    }
    const T &back() const {
        return buffer_[end_];
    }

    //Линеаризация - сдвинуть кольцевой буфер так, что его первый элемент
    //переместится в начало аллоцированной памяти. Возвращает указатель
    //на первый элемент.
    T *linearize() {
        if (begin_ == 0 || size_ == 0)
            return buffer_;

        T *temp = new T[size_];
        if (begin_ <= end_)
            memcpy(temp, buffer_ + begin_, size_ * sizeof(T));
        else {
            memcpy(temp, buffer_ + begin_, (capacity_ - begin_) * sizeof(T));
            memcpy(temp + (capacity_ - begin_), buffer_, (end_ + 1) * sizeof(T)); // <------Здесь
        }
        memcpy(buffer_, temp, size_ * sizeof(T));
        delete[](temp);
        begin_ = 0;
        end_ = size_ - 1;
        return buffer_;
    }
    //Проверяет, является ли буфер линеаризованным.
    [[nodiscard]] bool is_linearized() const {
        return (begin_ == 0);
    }
    //Сдвигает буфер так, что по нулевому индексу окажется элемент
    //с индексом new_begin.
    void rotate(int new_begin) {
        if (size_ == 0)
            return;

        if (size_ == capacity_) {
            begin_ = (begin_ + new_begin) % capacity_;
            end_ = (end_ + new_begin) % capacity_;
            return;
        }

        for (int i = 0; i < new_begin; i++) {
            buffer_[(end_ + 1 + i) % capacity_] = buffer_[(begin_ + i) % capacity_]; //Слишком просто чтобы быть правдой
        }
        begin_ = (begin_ + new_begin) % capacity_;
        end_ = (end_ + new_begin) % capacity_;
    }
    //Количество элементов, хранящихся в буфере.
    [[nodiscard]] int size() const {
        return size_;
    }
    [[nodiscard]] bool empty() const {
        return (size_ == 0);
    }
    //true, если size() == capacity().
    [[nodiscard]] bool full() const {
        return (size_ == capacity_);
    }
    //Количество свободных ячеек в буфере.
    [[nodiscard]] int reserve() const {
        return (capacity_ - size_);
    }
    //ёмкость буфера
    [[nodiscard]] int capacity() const {
        return capacity_;
    }

    void set_capacity(int new_capacity) {
        if (new_capacity == 0) {
            delete[](buffer_);
            begin_ = 0;
            end_ = 0;
            size_ = 0;
            capacity_ = 0;
            return;
        }

        if (new_capacity == capacity_)
            return;

        linearize();
        T *temp = new T[new_capacity];
        if (size_ <= new_capacity) {
            memcpy(temp, buffer_, size_ * sizeof(T));
            delete[](buffer_);
            buffer_ = temp;
            capacity_ = new_capacity;
        } else {
            memcpy(temp, buffer_, new_capacity * sizeof(T));
            delete[](buffer_);
            buffer_ = temp;
            capacity_ = new_capacity;
        }
    }

    //Изменяет размер буфера.
    //В случае расширения, новые элементы заполняются элементом item.
    void resize(int new_size, const T &item = T()) {
        if (new_size > capacity_)
            throw std::length_error("Size can't be greater than capacity.");

        if (new_size == size_)
            return;

        if (new_size < size_) {
            end_ = (begin_ + new_size - 1) % capacity_;
            size_ = new_size;
        } else {
            for (int i = 0; i < size_ - new_size; i++)
                buffer_[(end_ + 1 + i) % capacity_] = item;
            end_ = (begin_ + new_size - 1) % capacity_;
            size_ = new_size;
        }
    }

    //Оператор присваивания.
    CircularBuffer &operator=(const CircularBuffer &cb) {
        if (this == &cb)
            return *this;

        delete[](buffer_);
        buffer_ = new T[cb.capacity_];
        begin_ = cb.begin_;
        end_ = cb.begin_;
        size_ = cb.size_;
        capacity_ = cb.capacity_;
        for (int i = 0; i < size_; i++)
        {
            buffer_[(begin_ + i) % capacity_] = cb.buffer_[(begin_ + i) % capacity_];
        }
        return *this;
    }

    //Обменивает содержимое буфера с буфером cb.
    void swap(CircularBuffer &cb) {
        CircularBuffer temp = *this;
        *this = cb;
        cb = temp;
    }

    //Добавляет элемент в конец буфера.
    //Если текущий размер буфера равен его ёмкости, то переписывается
    //первый элемент буфера (т.е., буфер закольцован).
    void push_back(const T &item = T()) {
        if (capacity_ == 0)
            throw std::overflow_error("Can't push to buffer with zero capacity");
        if (size_ == 0)
        {
            buffer_[end_] = item;
            size_++;
            return;
        }
        if (size_ < capacity_) {
            end_ = (end_ + 1) % capacity_;
            buffer_[end_] = item;
            size_++;
        } else {
            begin_ = (begin_ + 1) % capacity_;
            end_ = (end_ + 1) % capacity_;
            buffer_[end_] = item;
        }
    }

    //Добавляет новый элемент перед первым элементом буфера.
    //Аналогично push_back, может переписать последний элемент буфера.
    void push_front(const T &item = T()) {
        if (capacity_ == 0)
            throw std::overflow_error("Can't push to buffer with zero capacity");

        if (size_ < capacity_) {
            begin_ = ((begin_ == 0) ? capacity_ - 1 : begin_ - 1);
            buffer_[begin_] = item;
            size_++;
        } else {
            begin_ = ((begin_ == 0) ? capacity_ - 1 : begin_ - 1);
            end_ = ((end_ == 0) ? capacity_ - 1 : end_ - 1);
            buffer_[begin_] = item;
        }
    }

    //удаляет последний элемент буфера.
    void pop_back() {
        end_ = ((end_ == 0) ? capacity_ - 1 : end_ - 1);
        size_--;
    }

    //удаляет первый элемент буфера.
    void pop_front() {
        begin_ = (begin_ + 1) % capacity_;
        size_--;
    }

    //Вставляет элемент item по индексу pos. Ёмкость буфера остается неизменной.
    void insert(int pos, const T &item = T()) {
        if (capacity_ == 0)
            throw std::overflow_error("Can't push to buffer with zero capacity");
        if (size_ == capacity_)
            throw std::overflow_error("Can't insert to full buffer");
        if (pos > size_)
            throw std::invalid_argument("Wrong Insert position");

        for (int i = 0; i < size_ - pos; i++)
            buffer_[(capacity_ + end_ - i + 1) % capacity_] = buffer_[(capacity_ + end_ - i) % capacity_];
        buffer_[pos] = item;
        size_++;
        end_ = (end_ + 1) % capacity_;
    }

    //Удаляет элементы из буфера в интервале [first, last).
    void erase(int first, int last) {
        int shift = last - first;
        for (int i = 0; i < shift; i++)
            buffer_[(first + i) % capacity_] = buffer_[(first + shift + i) % capacity_];
        size_ -= shift;
        end_ = (capacity_ + end_ - shift) % capacity_;
    }

    //Очищает буфер.
    void clear() {
        if (capacity_ == 0)
            return;
        delete[](buffer_);
        begin_ = 0;
        end_ = 0;
        size_ = 0;
        capacity_ = 0;
    }

    friend bool operator==(const CircularBuffer& a, const CircularBuffer& b) {
        if (&a == &b)
            return true;
        if (a.size_ != b.size_)
            return false;
        if (a.capacity_ != b.capacity_)
            return false;

        for (int i = 0; i < a.size_; i++) {
            if (a.buffer_[(a.begin_ + i) % a.capacity_] != b.buffer_[(b.begin_ + i) % b.capacity_])
                return false;
        }

        return true;
    }

    friend bool operator!=(const CircularBuffer &a, const CircularBuffer &b) {
        if (&a == &b)
            return false;
        if (a.size_ != b.size_)
            return true;
        if (a.capacity_ != b.capacity_)
            return true;
        for (int i = 0; i < a.size_; i++)
            if (a.buffer_[(a.begin_ + i) % a.capacity_] != b.buffer_[(b.begin_ + i) % b.capacity_])
                return true;

        return false;
    }
};
