import pandas as pd
import torch
import numpy as np
from anomaly_detector import AnomalyDetector

# 기존 설정 로드
detector = AnomalyDetector(
    model_path="autoencoder_model.pth",
    scaler_path="scaler.pkl",
    columns_path="columns.pkl",
    threshold=0.0 # 계산을 위해 임시로 0 설정
)

# 정상 샘플 로드
normal_df = pd.read_csv("normal_samples.csv")

# Loss 계산 함수
def get_losses(detector, df):
    losses = []
    # 전처리 (detector 내부 로직과 동일하게 수행)
    df_aligned = df.reindex(columns=detector.feature_columns, fill_value=0)
    df_aligned.replace([np.inf, -np.inf], 0, inplace=True)
    df_aligned.fillna(0, inplace=True)
    
    scaled_data = detector.scaler.transform(df_aligned.values)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    detector.model.eval()
    with torch.no_grad():
        reconstructed, _ = detector.model(input_tensor)
        # Loss 계산 (MSE)
        loss_val = torch.mean((input_tensor - reconstructed) ** 2, dim=1).numpy()
        losses.extend(loss_val)
    return np.array(losses)

# 정상 데이터에 대한 Loss 분포 확인
normal_losses = get_losses(detector, normal_df)

mean_loss = np.mean(normal_losses)
std_loss = np.std(normal_losses)
max_loss = np.max(normal_losses)

# 새로운 임계값 제안 (3-Sigma Rule)
new_threshold = mean_loss + (3 * std_loss)

print(f"=== Threshold Recalculation ===")
print(f"Min Loss : {np.min(normal_losses):.6f}")
print(f"Mean Loss: {mean_loss:.6f}")
print(f"Max Loss : {max_loss:.6f}")
print(f"Std Dev  : {std_loss:.6f}")
print(f"-------------------------------")
print(f"Current Threshold: 0.183441")
print(f"Suggested Threshold (Mean + 3*Std): {new_threshold:.6f}")
print(f"Conservative Threshold (Max * 1.1): {max_loss * 1.1:.6f}")
