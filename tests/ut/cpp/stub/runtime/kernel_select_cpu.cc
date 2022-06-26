#include <string>

namespace luojianet_ms {
namespace device {
namespace cpu {

bool IsDynamicParamKernel(const std::string &op_name) { return false; }

}  // namespace cpu
}  // namespace device
}  // namespace luojianet_ms
