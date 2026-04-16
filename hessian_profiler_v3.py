import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'ultralytics'))

import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from ultralytics.utils import DEFAULT_CFG

class HessianProfilerTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.hessian_traces = {}

    def run_hessian_trace(self, num_batches=3):
        """
        Sử dụng thuật toán Hutchinson để xấp xỉ Hessian Trace (Trace(H)).
        Layer nào có Trace(H) càng cao -> Độ nhạy cảm càng lớn -> Phải cấp nhiều bit (4-bit / 8-bit).
        Layer nào có Trace(H) thấp -> Khá lỳ đòn -> Ép thẳng xuống 1-bit.
        """
        print("\n[HESSIAN PROFILER] Khởi động mảng dò mìn đạn đạo Hutchinson...")
        
        # Thiết lập cơ sở DataLoader và Mạng (tận dụng code _setup_train gốc)
        self._setup_train()
        self.model.eval() # Không dùng BatchNorm tính running mean lúc dò mìn
        
        # Bắt đầu vòng lặp đo đạc trên vài batch
        dataloader_iter = iter(self.train_loader)
        
        # Tạo từ điển lưu dải nhạy cảm
        trace_dict = {name: 0.0 for name, param in self.model.named_parameters() if param.requires_grad}
        
        print(f"[HESSIAN PROFILER] Đang đo đạc trên {num_batches} batches...")
        for b in range(num_batches):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
                
            batch = self.preprocess_batch(batch)
            
            # Forward pass lấy Loss tổng
            preds = self.model(batch["img"])
            loss, _ = self.model.loss(batch, preds)
            loss = loss.sum()
            
            # Khởi tạo vector Rademacher v (random +1 hoặc -1) cho Hutchinson
            params = [p for p in self.model.parameters() if p.requires_grad]
            rademacher_v = [torch.randint_like(p, high=2, device=p.device) * 2 - 1.0 for p in params]
            
            # 1. Tính Gradient bậc 1: g = d(Loss) / d(Weight)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            
            # 2. Nhân v vô hướng với gradient (g^T * v)
            g_v = sum([torch.sum(g * v) for (g, v) in zip(grads, rademacher_v)])
            
            # 3. Tính Gradient bậc 2 (Hessian-vector product): Hv = d(g^T * v) / d(Weight)
            Hv = torch.autograd.grad(g_v, params, retain_graph=True)
            
            # 4. Tính toán Trace xấp xỉ = v^T * Hv
            i = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trace_estimate = torch.sum(rademacher_v[i] * Hv[i]).item()
                    trace_dict[name] += abs(trace_estimate) # Đo bình phương độ lệch
                    i += 1
            
            print(f"   + Hoàn thành Batch {b+1}/{num_batches}")
            
        # Chia trung bình trace
        for k in trace_dict.keys():
            trace_dict[k] /= num_batches
            
        return trace_dict

    def export_bit_assignment(self, trace_dict, output_yaml="bit_assignment.yaml"):
        """
        Dựa vào mức độ nhạy cảm (Hessian trace), tự động phân lô bit-width.
        Giả định: 
          - Nằm top 10% nhạy cảm nhất -> 8-bit
          - Nằm top 30% nhạy cảm tiếp theo -> 4-bit
          - Phần còn lại (Backbone cục súc) -> 1-bit
        """
        print("\n[HESSIAN PROFILER] Tính toán xong. Đang phân bổ tài nguyên Bit...")
        
        # Lọc ra các layer có trace (bỏ qua batchnorm/bias để tính gọn trọng số)
        conv_traces = {k: v for k, v in trace_dict.items() if 'weight' in k and 'bn' not in k}
        
        # Sắp xếp layer theo mức độ nhạy cảm (cao xuống thấp)
        sorted_layers = sorted(conv_traces.items(), key=lambda item: item[1], reverse=True)
        total_layers = len(sorted_layers)
        
        top10_idx = int(0.10 * total_layers)
        top40_idx = int(0.40 * total_layers)
        
        bit_assignment = {}
        for i, (layer_name, trace_val) in enumerate(sorted_layers):
            if i < top10_idx:
                bit = 8  # Lớp VVIP: Thường là Cổ (Neck) hoặc Stem
            elif i < top40_idx:
                bit = 4  # Lớp VIP: Các node quét Mamba 4 chiều
            else:
                bit = 1  # Lớp Công dân: Backbone xôi thịt
                
            bit_assignment[layer_name] = {'bit': bit, 'hessian_trace': float(trace_val)}
            
        with open(output_yaml, 'w') as f:
            yaml.dump(bit_assignment, f, default_flow_style=False)
            
        print(f"[HESSIAN PROFILER] 🎉 Chúc mừng! Sổ đỏ được cấp tại {output_yaml} \n")
        
def main():
    print("=== HFCAM-YOLO V3: HESSIAN-AWARE MIXED PRECISION (GIAI ĐOẠN 2) ===")
    
    # Cấu hình Override mồi để tải VisDrone và Model
    overrides = {
        'data': 'VisDrone.yaml',
        'epochs': 1, # Không train thực sự, chỉ tải mồi
        'imgsz': 640,
        'batch': 4,
        'device': 0,
        'model': 'yolov8-hfcam-visdrone.yaml'
    }
    
    profiler = HessianProfilerTrainer(overrides=overrides)
    
    # Kích hoạt dò mìn (3 batches là đủ có xu hướng)
    trace_matrix = profiler.run_hessian_trace(num_batches=3)
    
    # Xuất bản thiết kế Bit sang yaml
    profiler.export_bit_assignment(trace_matrix)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
