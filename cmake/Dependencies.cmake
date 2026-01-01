get_filename_component(QALSH4C_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party" ABSOLUTE)

add_library(CLI11::CLI11 INTERFACE IMPORTED GLOBAL)
target_include_directories(CLI11::CLI11 SYSTEM INTERFACE "${QALSH4C_THIRD_PARTY_DIR}/CLI11/include")

add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${QALSH4C_THIRD_PARTY_DIR}/eigen")

add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
target_include_directories(nlohmann_json::nlohmann_json SYSTEM INTERFACE
                           "${QALSH4C_THIRD_PARTY_DIR}/nlohmann_json/include")
