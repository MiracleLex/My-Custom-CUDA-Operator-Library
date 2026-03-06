from ._C import *

# 显式声明公开 API（可选，但推荐）
__all__ = [
    'vec_add',
    'vec_relu', 
    'vec_sigmoid',
    'vec_sum',
    'vec_softmax',
    'sgemm',
    'flash_attention',
    # 新增算子只需在这里添加一行
]