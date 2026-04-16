import sys
from pathlib import Path

# Trỏ đường dẫn vào thư viện ultralytics cục bộ đã được chỉnh sửa
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'ultralytics'))

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint

from ultralytics.utils import DEFAULT_CFG
from mixed_precision_v3 import apply_mixed_precision_from_yaml

# ==========================================
# PHASE 1 & 4: KDTrainer - Động Cơ Chưng Cất
# ==========================================
class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_model = None

    def setup_model(self):
        """Khớp nối Giai đoạn 3: Phẫu thuật Model ngay từ trong trứng nước"""
        ckpt = super().setup_model()
        
        # Gọi dao mổ 'Module Surgery' hoán đổi toàn bộ nn.Conv2d thành MPQ_Conv2d đa cấp bit
        print("\n[V3-ASSEMBLY] Chuẩn bị cấy ghép động cơ hỗn hợp Mixed Precision...")
        self.model = apply_mixed_precision_from_yaml(self.model, "bit_assignment.yaml")
        
        return ckpt

    def _setup_train(self):
        """Override hàm setup để nạp thêm mạng Teacher song song với mạng Student"""
        super()._setup_train()
        
        # 1. Tải trọng số mạng Teacher (YOLOv8n Full-Precision đã luyện 50 epochs)
        teacher_weights = r"C:\Users\dainhan\Desktop\PAPER\runs\detect\phase1_baseline\yolo_original_visdrone3\weights\best.pt"
        print(f"\n[KD-ENGINE] Đang khởi động mạng Teacher từ: {teacher_weights}")
        
        # 2. Khởi tạo Teacher (nạp trực tiếp module thay vì qua lớp YOLO wrapper để dễ bóc tách)
        self.teacher_model, _ = load_checkpoint(teacher_weights)
        self.teacher_model = self.teacher_model.to(self.device).eval() # Ép sang chế độ dự đoán (không dropout, etc.)
        
        # 3. Đóng băng vĩnh viễn (FREEZE) mạng Teacher để không tốn RAM tính đạo hàm
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        print("[KD-ENGINE] Teacher đã được nạp và đóng băng hoàn toàn. Mạch máu Gradient đã ngắt.")

    def forward(self, batch):
        """
        Nắn luồng Forward chuẩn của YOLO để lôi Teacher vào cuộc chơi.
        """
        # Bước 1: Cho mẻ ảnh (batch) chạy qua mạng Student như bình thường
        student_preds = self.model(batch["img"])
        student_loss, student_loss_items = self.model.loss(batch, student_preds)
        
        # Bước 2: Kích hoạt Teacher (Trạng thái tĩnh, không đạo hàm)
        with torch.no_grad():
            teacher_preds = self.teacher_model(batch["img"])
            
        # =============================================================
        # PHASE 4: ĐỘNG CƠ CHƯNG CẤT TRI THỨC (FEATURE-MAP MIMICKING)
        # =============================================================
        kd_loss = 0.0
        mse_criterion = nn.MSELoss()
        
        # YOLOv8 trả về các logit chưa nén dưới dạng list (mỗi phần tử đại diện cho 1 Scale phân giải: P3, P4, P5)
        # Chúng ta dùng MSE để ép Student phải học thuộc từng nếp gấp dao động cực nhỏ của Teacher!
        s_outputs = student_preds[0] if isinstance(student_preds, tuple) else student_preds
        t_outputs = teacher_preds[0] if isinstance(teacher_preds, tuple) else teacher_preds
        
        if isinstance(s_outputs, list) and isinstance(t_outputs, list):
            for s_map, t_map in zip(s_outputs, t_outputs):
                # [BẢO HIỂM]: Clamp ép trần/đáy trị số nội suy (tránh nổ FP16 AMP lúc nửa đêm)
                s_map_safe = torch.clamp(s_map, min=-20.0, max=20.0)
                t_map_safe = torch.clamp(t_map, min=-20.0, max=20.0)
                
                loss_kd_layer = mse_criterion(s_map_safe, t_map_safe)
                
                # Check tử hình: Nếu rủi ro có NaN lọt vào, vứt bỏ ngay không châm châm ngòi nổ
                if not torch.isnan(loss_kd_layer):
                    kd_loss += loss_kd_layer
                
        # Trọng số siêu tham số Distillation (alpha_kd).
        # Khuyên dùng: 0.1 để tránh sốc Gradient trong giai đoạn đầu
        alpha_kd = 0.1
        
        # Hợp thể: Tổng Loss = Học vẹt Label (student_loss) + Học mẹo logic không gian (KD)
        # Sử dụng torch.where để dự phòng student_loss tự nổ
        if torch.isnan(student_loss): 
            return kd_loss * 0.0, student_loss_items # Hủy lệnh cập nhật của batch này
            
        total_loss = student_loss + (alpha_kd * kd_loss)
        
        return total_loss, student_loss_items

# ==========================================
# KHU VỰC KHỞI CHẠY (Bấm Nút)
# ==========================================
def main():
    print("=== HFCAM-YOLO V3: KNOWLEDGE DISTILLATION INITIALIZATION ===")
    
    overrides = {
        'data': 'VisDrone.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 0,
        'name': 'yolo_hfcam_v3_kd_phase1',
        'project': 'runs/detect/phase3_hfcam',
        'model': 'yolov8-hfcam-visdrone.yaml',
        'optimizer': 'AdamW', # Sử dụng optimizer chuẩn tránh lỗi Muon Newton-Schulz
        'lr0': 0.001,         # Hạ Learning Rate cực đại xuống 1e-3 (Chuẩn AdamW)
        'warmup_epochs': 0,   # Tắt ép ga khởi động nhanh của YOLOv8 mặc định
        'warmup_bias_lr': 0.001, # Ngăn chặn nổ Bias
        'amp': False          # Ép chạy bằng chuẩn FP32 tuyệt đối, chặn đứng mọi lỗi tràn số (NaN)
    }
    
    trainer = KDTrainer(overrides=overrides)
    trainer.train()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
