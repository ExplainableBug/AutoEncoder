import shap
import torch
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

class AnomalyXAI:
    def __init__(self, detector, background_data, output_dir="xai_results"):
        """
        args:
            detector: 학습된 AnomalyDetector 인스턴스
            background_data: SHAP 기준점으로 사용할 정상 데이터 샘플 (DataFrame or Numpy array)
            output_dir: 결과 저장 경로
        """
        self.detector = detector
        self.output_dir = output_dir
        self.feature_columns = detector.feature_columns
        
        # SHAP 분석을 위한 배경 데이터 설정 (스케일링 적용)
        if isinstance(background_data, pd.DataFrame):
            # 컬럼 순서 보장 및 numpy 변환
            bg_data = background_data.reindex(columns=self.feature_columns, fill_value=0)
            self.bg_scaled = self.detector.scaler.transform(bg_data.values)
        else:
            self.bg_scaled = background_data # 이미 numpy라고 가정

        # 결과 저장 디렉토리 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # SHAP Explainer 초기화 (Loss 함수를 설명 대상으로 함)
        # link='identity'는 출력값을 그대로 사용함을 의미
        self.explainer = shap.KernelExplainer(
            self._predict_loss_wrapper, 
            self.bg_scaled
        )

    def _predict_loss_wrapper(self, x_numpy):
        """
        SHAP가 호출하는 함수. 
        입력 데이터(x_numpy)에 대해 모델의 MSE Loss를 반환함.
        """
        # Numpy -> Tensor
        input_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        
        self.detector.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.detector.model(input_tensor)
            # Row별 MSE Loss 계산: Mean((Input - Output)^2)
            # dim=1은 feature 차원을 의미
            loss = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
            
        return loss.numpy()

    def analyze_and_save(self, flow_data_dict, loss_val, partition, offset):
        """
        이상 데이터에 대해 SHAP 값을 계산하고 파일로 저장합니다.
        """
        try:
            # 1. 데이터 전처리 (detector 내부 로직과 동일하게 처리)
            df = pd.DataFrame([flow_data_dict])
            df = df.reindex(columns=self.feature_columns, fill_value=0)
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)
            
            # 스케일링
            x_scaled = self.detector.scaler.transform(df.values)

            # 2. SHAP 값 계산
            # nsamples는 추정 정확도와 속도 조절 (작을수록 빠름)
            shap_values = self.explainer.shap_values(x_scaled, nsamples=100)

            # 3. 결과 정리
            # shap_values는 리스트일 수 있으므로 처리 (Output 차원이 1개인 경우)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # 피처별 기여도 매핑
            contribution = dict(zip(self.feature_columns, shap_values[0].tolist()))
            
            # 기여도가 높은 순서로 정렬
            sorted_contribution = dict(sorted(contribution.items(), key=lambda item: item[1], reverse=True))

            # 4. JSON 저장
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "kafka_info": {"partition": partition, "offset": offset},
                "total_loss": float(loss_val),
                "threshold": self.detector.threshold,
                "shap_values": sorted_contribution,
                "raw_data": flow_data_dict
            }

            filename = f"anomaly_shap_p{partition}_o{offset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            print(f"[XAI] SHAP 분석 완료 및 저장됨: {filepath}")
            
            # 가장 기여도가 높은 Top 3 피처 반환 (로깅용)
            top_3_features = list(sorted_contribution.keys())[:3]
            return top_3_features

        except Exception as e:
            print(f"[XAI Error] 분석 실패: {e}")
            return []
