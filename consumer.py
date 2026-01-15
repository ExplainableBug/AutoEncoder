from kafka import KafkaConsumer
import json
import time
import os
import pandas as pd
from dotenv import load_dotenv

from anomaly_detector import AnomalyDetector
from xai_shap import AnomalyXAI

load_dotenv()

# --- 1. 환경 변수 ---
KAFKA_BROKER_ADDR = os.environ.get('KAFKA_BROKER_ADDR') 
KAFKA_BROKER_PORT = os.environ.get('KAFKA_BROKER1_PORT')
KAFKA_BROKER = f"{KAFKA_BROKER_ADDR}:{KAFKA_BROKER_PORT}"
TOPIC_NAME = os.environ.get('TOPIC_NAME')

# --- [설정] 이상 탐지기 초기화 ---
detector = AnomalyDetector(
    model_path="autoencoder_model.pth",
    scaler_path="scaler.pkl",
    columns_path="columns.pkl",
    threshold=0.183441
)

# --- [추가] XAI Explainer 초기화 ---
# 주의: SHAP 분석을 위해 '정상 데이터 샘플'이 필요합니다.
# 파일이 없으면 빈 DataFrame으로 시작하며 경고를 출력합니다.
xai_explainer = None
if os.path.exists("normal_samples.csv"):
    try:
        background_df = pd.read_csv("normal_samples.csv")
        # 계산 속도를 위해 샘플 수를 50개 정도로 제한하는 것이 좋습니다.
        if len(background_df) > 50:
            background_df = background_df.sample(50, random_state=42)
        
        xai_explainer = AnomalyXAI(detector, background_df)
        print("[System] XAI Explainer 초기화 완료")
    except Exception as e:
        print(f"[System Error] XAI 초기화 실패: {e}")
else:
    print("[Warning] 'normal_samples.csv'가 없어 XAI 기능을 사용할 수 없습니다.")




# --- 2. JSON Deserializer 함수 ---
def deserializeJson(m):
    try:
        # 바이트를 디코딩하고 JSON을 로드
        return json.loads(m.decode('utf-8'))
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류 발생: {e}. Raw Data: {m[:50]}...")
        return None

# --- 3. 메인 컨슈머 루프 ---
def startConsumer():
    # Kafka Consumer 설정
    
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='earlist', # 저장된 모든 메시지를 처음부터 확인
        enable_auto_commit=False, 
        group_id='group1',
        value_deserializer=deserializeJson 
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] 디버그 컨슈머 시작. 메시지를 기다립니다...")
    
    try:
        for message in consumer:
            flow_data = message.value
            print(f"[Raw Data]: {flow_data}\n")
            if flow_data is None: continue

            # 이상 탐지 수행
            is_anomaly, loss = detector.detect(flow_data)
            
            # 결과 출력 포맷팅
            status_msg = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
            
            print("-" * 50)
            print(f"Partition: {message.partition} | Offset: {message.offset} | {status_msg}")
            print(f"Loss  : {loss:.6f} (Threshold: {detector.threshold})")
	    
            if is_anomaly and xai_explainer is not None:
                print("   >> 이상 원인 분석(SHAP) 수행 중...")
                top_features = xai_explainer.analyze_and_save(
                    flow_data, 
                    loss, 
                    message.partition, 
                    message.offset
                )
                print(f"   >> 주요 원인 피처 Top 3: {top_features}")

#            메시지 내용이 너무 길면 일부만 출력하거나, 필요시 전체 출력
#            print(f"Data  : {flow_data}")

#            print(f"  Offset: {message.offset}")
#            print(f"  message : {message.value}")
            
    except KeyboardInterrupt:
        print("\n[INFO] 컨슈머 종료 요청 감지.")
    except Exception as e:
        print(f"[FATAL] 예외 발생: {e}")
    finally:
        consumer.close()
        print("[INFO] 컨슈머 연결이 닫혔습니다.")

if __name__ == '__main__':
    startConsumer()
