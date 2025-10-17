#pragma once

#include <cstring>
#include <fstream>
#include <vector>

class Buffer {
   public:
    Buffer() = default;
    void Init(unsigned int size);
    void Load(std::ifstream &ifs);
    void Save(std::ofstream &ofs) const;

    template <typename T>
    T Read(size_t &offset) const;

    template <typename T>
    std::vector<T> ReadVector(size_t &offset, size_t count) const;

    template <typename T>
    void Write(size_t &offset, const T &data);

   private:
    std::vector<char> data_;
};

template <typename T>
T Buffer::Read(size_t &offset) const {
    T data;
    std::memcpy(&data, &data_[offset], sizeof(T));
    offset += sizeof(T);

    return data;
}

template <typename T>
std::vector<T> Buffer::ReadVector(size_t &offset, size_t count) const {
    const size_t num_bytes_to_read = count * sizeof(T);
    std::vector<T> result(count);
    std::memcpy(result.data(), &data_[offset], num_bytes_to_read);

    offset += num_bytes_to_read;
    return result;
    return result;
}

template <typename T>
void Buffer::Write(size_t &offset, const T &data) {
    std::memcpy(&data_[offset], &data, sizeof(T));
    offset += sizeof(T);
}
