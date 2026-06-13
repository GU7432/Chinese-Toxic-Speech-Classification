import time
import torch

def test_pytorch_gpu():
    print("=" * 40)
    print(" PyTorch GPU 檢測工具")
    print("=" * 40)
    
    # 1. 檢查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA 核心是否可用: {cuda_available}")
    
    if not cuda_available:
        print("[-] 警告: 未偵測到 GPU，目前只能使用 CPU 進行運算。")
        print("    請檢查 PyTorch 版本是否正確（需為 +cu118 等 GPU 版本），以及 NVIDIA 驅動是否安裝。")
        return

    # 2. 顯示 GPU 相關資訊
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    cuda_version = torch.version.cuda
    
    print(f"[*] 偵測到的 GPU 數量: {device_count}")
    print(f"[*] 當前使用的 GPU 編號: {current_device}")
    print(f"[*] 顯卡型號: {device_name}")
    print(f"[*] PyTorch 編譯使用的 CUDA 版本: {cuda_version}")
    print("-" * 40)
    
    # 3. 實際將張量送入 GPU 進行運算測試
    print("[*] 正在進行 GPU 算力測試...")
    try:
        # 宣告使用 GPU 裝置
        device = torch.device("cuda")
        
        # 在 GPU 上建立兩個 5000x5000 的隨機矩陣
        start_time = time.time()
        x = torch.randn(5000, 5000, device=device)
        y = torch.randn(5000, 5000, device=device)
        
        # 執行矩陣乘法
        z = torch.matmul(x, y)
        
        # 確保運算完成
        torch.cuda.synchronize()
        end_time = time.time()
        
        print("[+] GPU 運算測試成功！")
        print(f"[+] 5000x5000 矩陣乘法耗時: {end_time - start_time:.4f} 秒")
        
    except Exception as e:
        print(f"[-] GPU 運算過程中發生錯誤: {e}")
        
    print("=" * 40)

if __name__ == "__main__":
    test_pytorch_gpu()