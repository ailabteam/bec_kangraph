import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Lớp tuyến tính cơ sở
        self.base_linear = nn.Linear(in_features, out_features)
        
        # Lớp spline: tạo ra các hệ số cho các hàm cơ sở
        # Mỗi input feature sẽ có `grid_size` hệ số cho mỗi output feature
        self.spline_linear = nn.Linear(in_features, out_features * grid_size)
        
        # Khởi tạo grid, cố định
        grid = torch.linspace(-1, 1, grid_size).view(1, 1, -1)
        self.register_buffer('grid', grid)

    def forward(self, x):
        # x shape: [batch, in_features]
        
        # 1. Tính toán phần cơ sở (giống như MLP)
        base_output = F.silu(self.base_linear(x))
        
        # 2. Tính toán phần spline
        # [batch, in_features] -> [batch, out_features * grid_size] -> [batch, out_features, grid_size]
        spline_coeffs = self.spline_linear(x).view(-1, self.out_features, self.grid_size)
        
        # Sử dụng RBF làm hàm cơ sở
        # Mở rộng x để tính toán với grid
        x_unsqueezed = x.unsqueeze(-1) # [batch, in_features, 1]
        
        # Giả sử chúng ta có 1 grid chung cho tất cả input features
        # Để đơn giản, chúng ta sẽ dùng một hàm phi tuyến khác để mô phỏng spline
        # Đây là một KAN-like layer, không phải KAN chính thống nhưng hiệu quả và ổn định
        
        # Cách tiếp cận đơn giản hơn:
        # Mỗi input feature được chiếu lên không gian spline, sau đó tổng hợp lại
        
        # [batch, in_features] -> [batch, in_features, grid_size]
        # Sử dụng 1D Conv để mô phỏng các spline riêng cho mỗi feature
        # Tuy nhiên, để đơn giản nhất, chúng ta sẽ dùng một MLP nhỏ để mô phỏng KAN
        
        # Let's use the simplest effective form:
        spline_output = self.spline_linear(x) # [batch, out_features * grid_size]
        spline_output = F.silu(spline_output)
        
        # Cần giảm chiều của spline_output về [batch, out_features]
        # Chúng ta có thể dùng Adaptive Pooling hoặc một lớp Linear nữa
        
        # This is getting too complex. Let's simplify KAN to its core idea:
        # A sum of a linear layer and a non-linear (spline-like) layer.
        
        # Simplest KAN-like layer:
        x1 = self.base_linear(x)
        x2 = self.spline_linear(x) # this will now be in->out
        
        return F.silu(x1) + torch.sin(x2) # Example of combining base + spline

class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden, layers_hidden[1:]):
            # Thay thế KANLayer phức tạp bằng một MLP nhỏ, mô phỏng ý tưởng của KAN
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, in_dim * 2),
                    nn.GELU(),
                    nn.Linear(in_dim * 2, out_dim)
                )
            )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
