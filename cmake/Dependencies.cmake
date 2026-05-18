include(FetchContent)

set(QALSH4C_CLI11_VERSION "v2.6.2")
set(QALSH4C_EIGEN_VERSION "5.0.1")
set(QALSH4C_NLOHMANN_JSON_VERSION "v3.12.0")

set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_INSTALL OFF CACHE BOOL "" FORCE)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    CLI11
    URL "https://github.com/CLIUtils/CLI11/archive/refs/tags/${QALSH4C_CLI11_VERSION}.tar.gz"
    URL_HASH SHA256=c6ea6b2e5608b3ea8617999bd5f47420c71b2ebdb8dc4767c1034d1da5785711
    SYSTEM
    EXCLUDE_FROM_ALL
)

FetchContent_Declare(
    nlohmann_json
    URL "https://github.com/nlohmann/json/releases/download/${QALSH4C_NLOHMANN_JSON_VERSION}/json.tar.xz"
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    SYSTEM
    EXCLUDE_FROM_ALL
)

FetchContent_Declare(
    eigen
    URL "https://gitlab.com/libeigen/eigen/-/archive/${QALSH4C_EIGEN_VERSION}/eigen-${QALSH4C_EIGEN_VERSION}.tar.gz"
    URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
    SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(CLI11 nlohmann_json eigen)

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()
