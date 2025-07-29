#ifndef SINK_H_
#define SINK_H_

#include <spdlog/common.h>
#include <spdlog/details/log_msg.h>

#include <cstdlib>
#include <memory>
#include <utility>

#include "spdlog/sinks/base_sink.h"

template <typename Mutex>
class TerminatingSink : public spdlog::sinks::base_sink<Mutex> {
   public:
    explicit TerminatingSink(std::shared_ptr<spdlog::sinks::sink> wrapped_sink);

   protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override;

   private:
    std::shared_ptr<spdlog::sinks::sink> wrapped_sink_;
};

template <typename Mutex>
TerminatingSink<Mutex>::TerminatingSink(std::shared_ptr<spdlog::sinks::sink> wrapped_sink)
    : wrapped_sink_(std::move(wrapped_sink)) {}

template <typename Mutex>
void TerminatingSink<Mutex>::sink_it_(const spdlog::details::log_msg& msg) {
    wrapped_sink_->log(msg);

    if (msg.level >= spdlog::level::err) {
        wrapped_sink_->flush();
        std::exit(EXIT_FAILURE);
    }
}

template <typename Mutex>
void TerminatingSink<Mutex>::flush_() {
    wrapped_sink_->flush();
}

#endif