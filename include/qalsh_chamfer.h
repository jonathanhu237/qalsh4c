#ifndef QALSH_CHAMFER_H_
#define QALSH_CHAMFER_H_

#include <memory>
#include <string>

namespace qalsh_chamfer {

class QalshChamfer;

class QalshChamferBuilder {
   public:
    QalshChamferBuilder() = default;

    auto set_dataset_name(const std::string& dataset_name) -> QalshChamferBuilder&;
    auto set_parent_directory(const std::string& parent_directory) -> QalshChamferBuilder&;
    auto set_num_points(unsigned int num_points) -> QalshChamferBuilder&;
    auto set_num_dimensions(unsigned int num_dimensions) -> QalshChamferBuilder&;
    auto set_verbose(bool verbose) -> QalshChamferBuilder&;

    [[nodiscard]] auto Build() const -> std::unique_ptr<QalshChamfer>;

   private:
    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    bool verbose_{false};
};

class QalshChamfer {
   public:
    auto PrintConfiguration() const -> void;
    auto Execute() const -> void;

    friend class QalshChamferBuilder;

   private:
    QalshChamfer(std::string dataset_name, std::string parent_directory, unsigned int num_points,
                 unsigned int num_dimensions, bool verbose);

    std::string dataset_name_;
    std::string parent_directory_;
    unsigned int num_points_{0};
    unsigned int num_dimensions_{0};
    bool verbose_{false};
};

}  // namespace qalsh_chamfer

#endif