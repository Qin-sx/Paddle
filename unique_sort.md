### PyTorch

#### 算子定义

pytorch/torch/functional.py

```python
def _unique_impl(
    input: Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None,
) -> _unique_impl_out:

    if has_torch_function_unary(input):
        return handle_torch_function(
            unique,
            (input,),
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    if dim is not None:
        output, inverse_indices, counts = _VF.unique_dim(
            input,
            dim,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    else:
        output, inverse_indices, counts = torch._unique2(
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    return output, inverse_indices, counts
```

#### 算子分发

pytorch/aten/src/ATen/native/native_functions.yaml

```yaml

- func: unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  variants: function
  dispatch:
    CPU: unique_dim_cpu
    CUDA: unique_dim_cuda
  tags: dynamic_output_shape
  autogen: unique_dim.out

- func: _unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  variants: function
  dispatch:
    CPU: _unique2_cpu
    CUDA: _unique2_cuda
    MPS: _unique2_mps
  tags: dynamic_output_shape
  autogen: _unique2.out

```

#### 算子实现

##### `cuda`

pytorch/aten/src/ATen/native/cuda/Unique.cu

```c++
std::tuple<Tensor, Tensor, Tensor>
_unique2_cuda(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return internal::unique_cuda_template<scalar_t>(self, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    return unique_dim_cuda_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}
```

##### `cpp`

PyTorch/pytorch/aten/src/ATen/native/Unique.cpp

```c++
std::tuple<Tensor, Tensor, Tensor>
_unique2_cpu(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  if (self.scalar_type() == kBool) {
    return unique_cpu_bool_template(self, return_inverse, return_counts);
  }
  return AT_DISPATCH_V2(self.scalar_type(), "unique", AT_WRAP([&] {
    // The current CPU implementation of unique always sort due to
    // this is faster than hash table
    return unique_cpu_sorted_template<scalar_t>(
        self, return_inverse, return_counts, IsUnique<scalar_t, /* equal_nan */ false>());
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_cpu(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_V2(self.scalar_type(), "unique_dim", AT_WRAP([&] {
    // The current implementation using `dim` always sorts due to unhashable tensors
    return _unique_dim_cpu_template<scalar_t>(self, dim, false, return_inverse, return_counts);
  }), AT_EXPAND(AT_ALL_TYPES), kBFloat16, kBool, kHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}
```

#### 测试代码

```python
import torch

def test_unique_sorted():

    x = torch.tensor([6, 4, 3, 1, 3, 4, 2, 1])
    print("Original tensor:", x)

    # sorted=True 的情况
    output_sorted = torch.unique(x, sorted=True)
    print("\nWith sorted=True:")
    print("Unique values (sorted):", output_sorted)  # 输出应该是 [1, 2, 3, 4, 6]

    # sorted=False 的情况
    output_unsorted = torch.unique(x, sorted=False)
    print("\nWith sorted=False:")
    print("Unique values (unsorted):", output_unsorted)  # 输出可能是 [6, 4, 3, 1, 2]

    output_full_sorted, indices_sorted, counts_sorted = torch.unique(
        x,
        sorted=True,
        return_inverse=True,
        return_counts=True
    )
    print("\nWith sorted=True and full return:")
    print("Unique values:", output_full_sorted)
    print("Inverse indices:", indices_sorted)
    print("Counts:", counts_sorted)

    output_full_unsorted, indices_unsorted, counts_unsorted = torch.unique(
        x,
        sorted=False,
        return_inverse=True,
        return_counts=True
    )
    print("\nWith sorted=False and full return:")
    print("Unique values:", output_full_unsorted)
    print("Inverse indices:", indices_unsorted)
    print("Counts:", counts_unsorted)

if __name__ == "__main__":
    test_unique_sorted()
```

```bash
Original tensor: tensor([6, 4, 3, 1, 3, 4, 2, 1])

With sorted=True:
Unique values (sorted): tensor([1, 2, 3, 4, 6])

With sorted=False:
Unique values (unsorted): tensor([1, 2, 3, 4, 6])

With sorted=True and full return:
Unique values: tensor([1, 2, 3, 4, 6])
Inverse indices: tensor([4, 3, 2, 0, 2, 3, 1, 0])
Counts: tensor([2, 1, 2, 2, 1])

With sorted=False and full return:
Unique values: tensor([1, 2, 3, 4, 6])
Inverse indices: tensor([4, 3, 2, 0, 2, 3, 1, 0])
Counts: tensor([2, 1, 2, 2, 1])
```

### 总结

PyTorch的[unique函数](https://pytorch.org/docs/2.5/generated/torch.unique.html#torch.unique)的sorted参数并没有使用。
1. 在`cpp`和`cuda`具体实现中并没有传入和使用sorted参数。
2. python代码设置sorted参数为true和false没有区别。
