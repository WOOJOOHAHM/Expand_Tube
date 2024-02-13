import torch
def main():
    # 현재 사용 가능한 GPU 개수 확인
    num_devices = torch.cuda.device_count()

    # 각 GPU의 메모리를 해제
    for device_idx in range(num_devices):
        torch.cuda.empty_cache()

if __name__ == '__main__': main()