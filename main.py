import polars as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model.incremental import IncrementalAutoCox

def generate_synthetic_data(n_samples=200, n_old=10, n_new=10):
    """生成模拟的生存数据用于演示"""
    np.random.seed(42)
    
    # 生成特征
    X_old = np.random.randn(n_samples, n_old)
    X_new = np.random.randn(n_samples, n_new)
    
    # 模拟生存时间 (基于特征的线性组合 + 噪声)
    risk_score = 0.5 * X_old[:, 0] + 0.8 * X_new[:, 0] + np.random.randn(n_samples) * 0.2
    time = np.exp(-risk_score) * 1000  # 天数
    event = np.random.binomial(1, 0.7, n_samples) # 70% 发生事件
    
    # 构造带正负号的标签
    y = np.where(event == 1, time, -time).reshape(-1, 1)
    
    return X_old.astype(np.float32), X_new.astype(np.float32), y.astype(np.float32)

# ==========================================
# 步骤 1: 模拟预训练阶段
# ==========================================
print("--- 阶段 1: 预训练 Base 模型 (仅旧特征) ---")
X_old_pre, _, y_pre = generate_synthetic_data(n_samples=150, n_old=10)

base_config = {
    'n_input': 10, 
    'n_hidden_1': 32, 
    'n_hidden_2': 16, 
    'L2_reg': 0.01,
    'dropout': 0.1
}

# 初始化 Base 模型
base_model = IncrementalAutoCox(hidden_layers_nodes=[8, 1], encoder_config=base_config)
# 训练 Base 模型
base_model.fit(X_old_pre, y_pre, num_steps=100, silent=True)
print("Base 模型训练完成。\n")

# ==========================================
# 步骤 2: 模拟 2021 年的增量学习阶段
# ==========================================
print("--- 阶段 2: 增量学习 WMDX (旧特征 + 新特征) ---")
X_old_inc, X_new_inc, y_inc = generate_synthetic_data(n_samples=100, n_old=10, n_new=10)
X_combined = np.hstack([X_old_inc, X_new_inc])

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_inc, test_size=0.2, random_state=42)

inc_config = {
    'n_input': 20,
    'feature_split': [10, 10], # 10个旧特征，10个新特征
    'n_hidden_1': 32, 
    'n_hidden_2': 16,
    'old_lr_ratio': 0.0,      # 冻结旧知识
    'new_lr_ratio': 0.05,     # 缓慢学习新知识
    'warm_up_epoch': 30, 
    'loss_alpha': 0.7, 
    'loss_alpha2': 0.3,            
}

inc_model = IncrementalAutoCox(hidden_layers_nodes=[8, 1], encoder_config=inc_config)

# --- 权重移植 ---
with torch.no_grad():
    inc_model.encoder_old_dense.weight.copy_(base_model.encoder_dense_1.weight)
    inc_model.decoder_old.weight.copy_(base_model.decoder_dense.weight)
    # 移植 Risk Head
    for i, layer in enumerate(base_model.risk_head):
        if isinstance(layer, nn.Linear):
            inc_model.risk_head[i].weight.copy_(layer.weight)

# --- 开始增量训练并自动触发全指标评估 ---
history = inc_model.fit(
    data_X=X_train,
    data_y=y_train,
    test_x=X_test,
    test_y=y_test,
    num_steps=200,
    num_skip_steps=50,
    history_path="demo_training_history.json"
)

# ==========================================
# 步骤 3: 结果展示
# ==========================================
print("\n增量学习最终临床指标摘要:")
metrics = history["final_clinical_metrics"]
print(f"- IPCW C-index: {metrics.get('IPCW_C_Index', 0):.4f}")
print(f"- Overall IBS: {metrics.get('IBS_Overall', 0):.4f}")

# 预测生存概率示例 (Polars 输出)
surv_probs = inc_model.predict_survival_function(X_test[:5], times=[100, 300, 500])
print("\n前 5 个样本在特定时间点的生存概率预测 (Polars):")
print(surv_probs)

print("\n示例运行成功！请查看生成的 demo_training_history.json 获取全量指标。")