如果前向网络使用了SparseGatherV2等算子，优化器会执行稀疏运算，通过设置 `target` 为CPU，可在主机（host）上进行稀疏运算。
稀疏特性在持续开发中。
