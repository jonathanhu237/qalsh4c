#include "buffer.h"

void Buffer::Init(unsigned int size) { data_.resize(size); }

void Buffer::Load(std::ifstream &ifs) { ifs.read(data_.data(), static_cast<std::streamsize>(data_.size())); }

void Buffer::Save(std::ofstream &ofs) const { ofs.write(data_.data(), static_cast<std::streamsize>(data_.size())); }