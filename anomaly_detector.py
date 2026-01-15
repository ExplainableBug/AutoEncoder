import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. Autoencoder 모델 클래스 (학습 코드와 동일해야 함)
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ==========================================
# 2. 이상 탐지기 클래스 (외부에서 사용할 인터페이스)
# ==========================================
class AnomalyDetector:
    def __init__(self, model_path, scaler_path, columns_path, threshold):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.columns_path = columns_path
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self._load_resources()

    def _load_resources(self):
        """필요한 파일(pth, pkl)을 로드합니다."""
        if not (os.path.exists(self.scaler_path) and os.path.exists(self.columns_path) and os.path.exists(self.model_path)):
            print(f"[Detector Error] 필수 파일이 누락되었습니다. 경로를 확인하세요.")
            return

        try:
            # 1. 컬럼 및 스케일러 로드
            self.feature_columns = joblib.load(self.columns_path)
            self.scaler = joblib.load(self.scaler_path)
            input_dim = len(self.feature_columns)

            # 2. 모델 로드
            self.model = Autoencoder(input_dim=input_dim, latent_dim=4)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            print(f"[Detector] 모델 로드 완료 (Threshold: {self.threshold})")
            
        except Exception as e:
            print(f"[Detector Error] 리소스 로드 중 오류 발생: {e}")

    def detect(self, flow_data_dict):
        """
        단일 Flow 데이터(Dict)를 받아 이상 여부를 반환합니다.
        Returns: (is_anomaly: bool, loss: float)
        """
        if self.model is None or self.scaler is None:
            return False, 0.0

        try:
            # 1. Dict -> DataFrame
            df = pd.DataFrame([flow_data_dict])
            
            # 2. 컬럼 순서 맞추기 (없는 컬럼은 0으로 채움)
            df_features = df.reindex(columns=self.feature_columns, fill_value=0)
            drop_cols = ['filename', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol']

            for col in drop_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # --- [디버깅 코드 시작] ---
            # 실제로 데이터가 들어오는지, 아니면 모두 0인지 확인
            # 확인 후 주석 처리 필요
            print(f"[DEBUG] Input columns: {list(df.columns)}")
            print(f"[DEBUG] Model columns: {self.feature_columns[:5]} ...")
            print(f"[DEBUG] DataFrame Values Head: {df_features.iloc[0].values[:10]}")
            # --- [디버깅 코드 끝] ---
            
            # 3. 결측치/무한대 처리
            df_features.replace([np.inf, -np.inf], 0, inplace=True)
            df_features.fillna(0, inplace=True)
            
            # 4. 스케일링
            X_scaled = self.scaler.transform(df_features.values)
            # [추가] 스케일링된 값이 0~1 사이인지, 아니면 10, 100 처럼 튀는지 확인
            print(f"[DEBUG] Scaled Max: {X_scaled.max():.4f} | Min: {X_scaled.min():.4f}")
            input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # 5. 추론
            with torch.no_grad():
                reconstructed, _ = self.model(input_tensor)
                loss = torch.mean((input_tensor - reconstructed) ** 2, dim=1).item()
            
            is_anomaly = loss > self.threshold	
            return is_anomaly, loss
            
        except Exception as e:
            print(f"[Detector Error] 추론 중 오류: {e}")
            return False, 0.0
