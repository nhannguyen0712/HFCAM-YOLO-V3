import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path

# ==========================================
# PHASE 3: LÕI TOÁN HỌC MIXED-PRECISION
# ==========================================

class MixedPrecisionSTE(torch.autograd.Function):
    """
    Động cơ ép bit tuỳ chỉnh (Heterogeneous Quantization Engine)
    Sử dụng Straight-Through Estimator (STE) truyền gradient qua bước sóng không khả vi.
    """
    @staticmethod
    def forward(ctx, w, num_bits):
        ctx.save_for_backward(w)
        ctx.num_bits = num_bits

        if num_bits == 1:
            # =======================================
            # Chế độ 1-bit: Binarization Truyền thống
            # Kích hoạt khi Trace Hessian = Rất Thấp
            # =======================================
            w_b = torch.sign(w)
            # Ép phần tử 0 lùi về 1 để không bị triệt tiêu
            w_b[w_b == 0] = 1.0
            return w_b
            
        elif num_bits == 32:
            return w # Giữ nguyên Full-Precision (Thường dành cho cực kỳ nhạy cảm)

        else:
            # =======================================
            # Chế độ 4-bit / 8-bit: LSQ / MinMax DoReFa
            # Kích hoạt khi Trace Hessian = Cực kỳ Lớn (Vùng Mamba/Head)
            # =======================================
            Qn = -(2 ** (num_bits - 1))      # VD 4-bit: -8
            Qp = (2 ** (num_bits - 1)) - 1   # VD 4-bit: 7
            
            # Tính Step-Size động dựa trên năng lượng Max của Weight
            # (Có thể nâng cấp thành Learnable Step Size cho bản V4)
            w_abs_max = torch.max(torch.abs(w))
            scale = w_abs_max / Qp
            scale = torch.clamp(scale, min=1e-8) # Cứu mạng cho Gradient nổ (NaN loss)
            
            # Quantize - Ép khung 16 bậc thang (4-bit) hoặc 256 bậc (8-bit)
            w_q = torch.round(w / scale)
            w_q = torch.clamp(w_q, Qn, Qp)
            
            # Dequantize - Trả lại mức biên độ thật để Loss Function không bị sốc
            w_dq = w_q * scale
            return w_dq

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-Through Estimator: Grad_in = Grad_out"""
        w, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # Hard-Tanh Clipping: Cắt bỏ gradient các weight vượt rào quá gắt 
        # (Chỉ cắt cho 1-bit, còn multi-bit thì lưới mịn hơn nên nới lỏng)
        if ctx.num_bits == 1:
            grad_input[w.abs() > 1.0] = 0 
            
        return grad_input, None

class MPQ_Conv2d(nn.Conv2d):
    """
    Lớp Tích chập đa hình thể (Mixed Precision Quantized Convolution)
    Có thể tự biến hình thành 1-bit XNOR, 4-bit Shift-Add, hoặc 8-bit tuỳ biến config.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bits = num_bits # Số bit mặc định, sẽ bị override khi load yaml

    def forward(self, x):
        # 1. Gọi lõi ép bit đa hình thể lên mảng Trọng số
        quantized_weight = MixedPrecisionSTE.apply(self.weight, self.num_bits)
        
        # 2. (Tuyệt Kỹ V3): Khác ở chỗ ép bit linh hoạt, ta tiến lên Conv2d tiêu chuẩn
        # Note: Ở bản triển khai FPGA thực tế, ta sẽ đổi hàm F.conv2d này bằng code POPCNT/BitShift phần cứng
        out = F.conv2d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

def apply_mixed_precision_from_yaml(base_model, yaml_path="bit_assignment.yaml"):
    """
    Hàm quyền lực (Surgical Module): Tưới cấu hình Hessian Bit xuống mọi bó cơ của Mamba và YOLO.
    Bằng kỹ thuật 'Hot Swap', thay thế tất cả nn.Conv2d thông thường thành MPQ_Conv2d động cơ lượng tử!
    """
    assignment = {}
    if Path(yaml_path).exists():
        with open(yaml_path, 'r') as f:
            assignment = yaml.safe_load(f)
            print(f"\n[MIXED-PRECISION] Nạp thành công sổ đỏ phân vùng từ '{yaml_path}'")
    else:
        print("\n[MIXED-PRECISION] KHÔNG TÌM THẤY SỔ ĐỎ HESSIAN. Chạy mặc định 1-bit toàn tuyến!")

    # Hàm quy hồi phẫu thuật nơ-ron
    def replace_conv(module, prefix=""):
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            # Gặp tế bào Conv2d -> Tiến hành hóa kiếp
            if isinstance(child, nn.Conv2d) and not isinstance(child, MPQ_Conv2d):
                allocated_bits = 1 # Mặc định nhà nghèo 1-bit
                
                # Nếu được sổ đỏ độ nhạy cảm rót vốn
                weight_key = f"{full_name}.weight"
                if weight_key in assignment:
                    allocated_bits = assignment[weight_key]['bit']
                
                # Sinh ra tế bào lượng tử mới
                new_conv = MPQ_Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    bias=(child.bias is not None), num_bits=allocated_bits
                )
                
                # Bơm lại máu (Weights gốc) sang tim mới
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                    
                # Hoán đổi linh hồn
                setattr(module, child_name, new_conv)
            else:
                # Tiếp tục đào sâu vào các Block con (C2f, Bottleneck, MambaBlock...)
                replace_conv(child, full_name)
                
    # Bắt đầu phẫu thuật toàn thân mô hình
    neural_net = base_model.model if hasattr(base_model, 'model') else base_model
    replace_conv(neural_net, prefix="model")
    
    print("[MIXED-PRECISION] Hoàn thành Cấy Ghép Lượng tử. Tất cả Conv2d đã bị thay thế thành MPQ_Conv2d.")
    return base_model
