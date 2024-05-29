import os
import sys
import pandas as pd
import numpy as np

# 디렉토리 설정 및 모듈 경로 추가
PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import matplotlib.pyplot as plt
from config import get_config
from _commons.utils import load_fcst_y

def load_node_indices(topic_dir):
    """
    주어진 토픽 디렉토리에서 node indices 정보를 로드합니다.
    :param topic_dir: 토픽 디렉토리 경로
    :return: node indices 데이터프레임
    """
    tsv_file_path = os.path.join(topic_dir, "node_indices.tsv")
    df = pd.read_csv(tsv_file_path, sep='\t', header=None)
    df.columns = ['index', 'keyword']  # 컬럼 이름 설정
    return df

def save_predictions_to_tsv(node_indices_df, predictions, model_name, output_dir):
    """
    예측 결과를 키워드별로 TSV 파일로 저장합니다.
    :param node_indices_df: node indices 데이터프레임
    :param predictions: 모델 예측 결과
    :param model_name: 모델 이름
    :param output_dir: 출력 디렉토리 경로
    """
    # 각 키워드별 예측값을 데이터프레임으로 변환합니다.
    results = []
    for i, row in node_indices_df.iterrows():
        keyword = row['keyword']
        pred_values = predictions[:, :, i].flatten()  # 모든 배치와 예측 시간을 포함한 예측값
        for time_step, value in enumerate(pred_values):
            results.append([keyword, time_step, value])

    results_df = pd.DataFrame(results, columns=['keyword', 'time_step', f'{model_name}_prediction'])
    output_path = os.path.join(output_dir, f'{model_name}_predictions.tsv')
    results_df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    # 설정 로드
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    fcst_wc_save_path = os.path.join(results_path, 'fcst_wc')

    node_feature_type = '_'.join(args.best_node_feature_type)
    # 예측값 로드
    true_ys, fcst_ys_lstm, xs = load_fcst_y(fcst_val_save_path, args, 'lstm', load_x=True)
    _, fcst_ys_gru, _ = load_fcst_y(fcst_val_save_path, args, 'gru', load_x=True)
    _, fcst_ys_agcrn, _ = load_fcst_y(fcst_val_save_path, args, 'agcrn', load_x=True)
    _, fcst_ys_a3tgcn2, _ = load_fcst_y(fcst_val_save_path, args, 'a3tgcn2', load_x=True)

    # 첫 번째 토픽의 node indices 로드
    topic_num = 1
    topic_dir = os.path.join(args.topic_dir, str(topic_num))  # 실제 토픽 디렉토리 경로로 변경해야 합니다.
    node_indices_df = load_node_indices(topic_dir)

    # 예측 결과 저장
    save_predictions_to_tsv(node_indices_df, fcst_ys_gru, 'gru', fcst_wc_save_path)
    save_predictions_to_tsv(node_indices_df, fcst_ys_lstm, 'lstm', fcst_wc_save_path)
    save_predictions_to_tsv(node_indices_df, fcst_ys_agcrn, 'agcrn', fcst_wc_save_path)
    save_predictions_to_tsv(node_indices_df, fcst_ys_a3tgcn2, 'a3tgcn2', fcst_wc_save_path)

    print(f"Predictions saved to {fcst_wc_save_path}")
