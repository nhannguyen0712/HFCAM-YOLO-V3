import sys
from pathlib import Path

# Trỏ đường dẫn vào thư viện ultralytics cục bộ
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'ultralytics'))

from train_v3_kd import KDTrainer

def get_latest_checkpoint():
    """Tự động săn tìm hòm chứa bí mật (best.pt) lưu gần nhất để cách ly thảm họa NaN"""
    runs_dir = current_dir / "runs"
    
    if not runs_dir.exists():
        print("❌ LỖI: Không tìm thấy thư mục 'runs'.")
        return None
        
    # Quét đệ quy tìm tất cả các file 'last.pt' để quy ra thư mục đó
    checkpoints = list(runs_dir.rglob("last.pt"))
    
    if not checkpoints:
        return None
        
    latest_last_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    # [QUAN TRỌNG]: Đổi đuôi last.pt sang best.pt để loại trừ vòng lặp Epoch cuối bị sập (Corrupted Checkpoint)
    best_ckpt = latest_last_ckpt.parent / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    return latest_last_ckpt

def main():
    print("\n" + "="*50)
    print("🚀 HFCAM-YOLO V3: KHỞI ĐỘNG LẠI (RESUME) 🚀")
    print("="*50)
    
    ckpt_path = get_latest_checkpoint()
    if ckpt_path is None:
        return
        
    print(f"\n[RESUME] Chặn cửa tử thần! Vượt mặt Epoch gãy, quay lùi về Trạm an toàn nhất tại:\n 👉 {ckpt_path.relative_to(current_dir)}")
    
    overrides = {
        'model': str(ckpt_path),
        'resume': True,
        'amp': False # Ép buộc chuyển chế pháo đài FP32
    }
    
    # Khởi xướng KDTrainer với khối kiến thức bị cắt ngang
    trainer = KDTrainer(overrides=overrides)
    trainer.train()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
