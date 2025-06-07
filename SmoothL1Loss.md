## SmoothL1Loss

[paddle文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SmoothL1Loss_cn.html)
[pytorch文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

### paddle

$$ z_i = 0.5 * (x_i - y_i)^2, if |x_i - y_i| < delta $$
$$ z_i = delta * |x_i - y_i| - 0.5 * delta^2, otherwise $$

### pytorch

$$ z_i = 0.5 * (x_i - y_i)^2 / delta, if |x_i - y_i| < delta $$
$$ z_i = |x_i - y_i| - 0.5 * delta, otherwise $$

### 区别

paddle应该是huber(x,y)
pytorch应该是huber(x,y)/beta

### 测试代码

```python
import paddle
import torch
import numpy as np

np.random.seed(42)
paddle.seed(42)
torch.manual_seed(42)
delta = 0.7

input_np = np.random.rand(3, 3).astype(np.float32)
label_np = np.random.rand(3, 3).astype(np.float32)

input_paddle = paddle.to_tensor(input_np)
label_paddle = paddle.to_tensor(label_np)
input_torch = torch.tensor(input_np)
label_torch = torch.tensor(label_np)

# Paddle SmoothL1Loss
paddle_loss = paddle.nn.SmoothL1Loss(delta=delta)
paddle_output = paddle_loss(input_paddle, label_paddle)

# PyTorch SmoothL1Loss
torch_loss = torch.nn.SmoothL1Loss(beta=delta)
torch_output = torch_loss(input_torch, label_torch)

print("NumPy 生成的输入:")
print("Input:")
print(input_np)
print("\nLabel:")
print(label_np)

print("\nPaddle 结果:", paddle_output.item())
print("PyTorch 结果:", torch_output.item())
print("差异:", abs(paddle_output.item() - torch_output.item()))

diff_np = input_np - label_np
abs_diff = np.abs(diff_np)
smooth_l1_values = np.where(abs_diff < delta,
                           0.5 * abs_diff * abs_diff,
                           delta * abs_diff - 0.5 * delta * delta)
# smooth_l1_values = np.where(abs_diff < delta,
#                            0.5 * abs_diff * abs_diff / delta,
#                            abs_diff - 0.5 * delta)
manual_result = np.mean(smooth_l1_values)

print("\n手动计算结果:", manual_result)
print("与 Paddle 差异:", abs(manual_result - paddle_output.item()))
print("与 PyTorch 差异:", abs(manual_result - torch_output.item()))

```
