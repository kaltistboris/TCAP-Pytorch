import numpy as np
import pandas as pd
import warnings
from scipy.stats import chi2
# 引入 sksurv 相关的评估库和数据格式化工具
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
    brier_score
)
from sksurv.nonparametric import kaplan_meier_estimator

def _check_config(config):
    """
    检查配置并用默认配置补全
    """
    default_config = {
        "learning_rate": 0.001,
        "learning_rate_decay": 1.0,
        "activation": "relu",  # 配合你网络层的 relu
        "L2_reg": 0.0,
        "L1_reg": 0.0,
        "optimizer": "adam",
        "dropout_keep_prob": 1.0,
        "seed": 42
    }
    for k in default_config.keys():
        if k not in config:
            config[k] = default_config[k]

def _check_surv_data(surv_data_X, surv_data_y):
    """检查生存数据类型的合法性"""
    if not isinstance(surv_data_X, pd.DataFrame):
        raise TypeError("The type of X must be DataFrame.")
    if not isinstance(surv_data_y, pd.DataFrame) or len(surv_data_y.columns) != 1:
        raise TypeError("The type of y must be DataFrame and contains only one column.")

def _prepare_surv_data(surv_data_X, surv_data_y):
    """
    生存数据准备：根据绝对时间(T)降序排列。
    这是 Breslow 近似计算偏似然函数的重要前置条件。
    """
    _check_surv_data(surv_data_X, surv_data_y)
    T = -np.abs(np.squeeze(np.array(surv_data_y)))
    sorted_idx = np.argsort(T)
    return sorted_idx, surv_data_X.iloc[sorted_idx, :], surv_data_y.iloc[sorted_idx, :]

def _to_sksurv_format(y):
    """
    将带有正负号的生存标签转换为 scikit-survival 要求的 Structured Array
    y: array-like, >0 表示事件发生(死亡/复发)，<0 表示右删失，绝对值为发生时间
    """
    y = np.squeeze(np.asarray(y))
    time = np.abs(y)
    event = (y > 0).astype(bool)
    # 使用官方工具确保格式严格合规
    return Surv.from_arrays(event=event, time=time)

def _baseline_hazard(label_e, label_t, pred_hr):
    """
    内部方法：计算基线危险率 (Baseline Hazard)
    """
    ind_df = pd.DataFrame({"E": label_e, "T": label_t, "P": pred_hr})
    # 按发生时间 T 分组，求该时间点的事件总数 E，以及风险集的风险指数和 P
    summed_over_durations = ind_df.groupby("T")[["P", "E"]].sum()
    
    # Breslow 近似公式的分母：当前时间点风险集内所有个体的风险比 (HR) 之和
    # 利用 pandas 倒序 cumsum 来快速计算 T >= t 的样本集合
    reversed_cumsum = summed_over_durations["P"].iloc[::-1].cumsum()
    summed_over_durations["P"] = reversed_cumsum
    
    base_haz = pd.DataFrame(
        summed_over_durations["E"] / summed_over_durations["P"], columns=["base_haz"]
    )
    return base_haz

def _baseline_cumulative_hazard(label_e, label_t, pred_hr):
    """计算基线累积危险率"""
    return _baseline_hazard(label_e, label_t, pred_hr).cumsum()

def baseline_survival_function(y, pred_hr):
    """
    通过 Breslow 估计法计算基线生存函数 S0(t)
    
    Parameters
    ----------
    y : np.array
        带正负号的生存标签
    pred_hr : np.array
        模型预测的危险比 (Hazard Ratio = exp(risk_score))
        
    Returns
    -------
    DataFrame
        Index 为发生事件的时间点，Column 为对应的基线生存概率。
    """
    y = np.squeeze(np.asarray(y))
    pred_hr = np.squeeze(np.asarray(pred_hr))
    
    t = np.abs(y)
    e = (y > 0).astype(np.int32)
    
    base_cum_haz = _baseline_cumulative_hazard(e, t, pred_hr)
    survival_df = np.exp(-base_cum_haz)
    return survival_df

def predict_surv_prob(BSF, pred_hr, times):
    """
    【新增核心功能】根据基线生存函数预测个体在特定时间点的生存概率
    
    Parameters
    ----------
    BSF : DataFrame
        通过 baseline_survival_function 计算出的基线生存函数
    pred_hr : np.array
        个体预测的风险比 (Hazard Ratio)
    times : list or np.array
        需要评估的时间点数组 (例如 [3, 6, 12])
        
    Returns
    -------
    np.ndarray
        返回生存概率矩阵，Shape 为 (n_samples, n_times)
    """
    base_times = BSF.index.values
    base_surv = np.squeeze(BSF.values)
    
    # 补充 t=0 时的生存率为 1.0，确保插值合法
    if base_times[0] > 0:
        base_times = np.insert(base_times, 0, 0.0)
        base_surv = np.insert(base_surv, 0, 1.0)
        
    # 生存函数是阶跃函数，使用 searchsorted 寻找对于任意时间点 t 适用的最近基线生存概率
    idx = np.searchsorted(base_times, times, side='right') - 1
    idx = np.clip(idx, 0, len(base_times) - 1)
    target_base_surv = base_surv[idx]  # Shape: (n_times,)
    
    # 个体生存概率推导公式: S(t|x) = S0(t) ^ exp(beta * X)
    pred_hr = np.asarray(pred_hr).reshape(-1, 1)
    target_base_surv = target_base_surv.reshape(1, -1)
    
    # 矩阵广播操作，得到 (n_samples, n_times) 的二维矩阵
    surv_prob_matrix = np.power(target_base_surv, pred_hr)
    
    return surv_prob_matrix

def eval_c_index(y_test, risk_scores):
    """
    计算标准的 Harrell's C-index
    
    Parameters
    ----------
    y_test : np.array
        测试集生存标签 (带正负号)
    risk_scores : np.array
        模型预测的风险分数 (通常为 log_HR 或 HR，值越大代表风险越高)
        
    Returns
    -------
    float
        Harrell's C-index
    """
    y_test_surv = _to_sksurv_format(y_test)
    
    # 返回值为 (c_index, concordant, discordant, tied_risk, tied_time)
    c_index = concordance_index_censored(
        y_test_surv["event"], 
        y_test_surv["time"], 
        np.squeeze(risk_scores)
    )[0]
    
    return c_index

def eval_ipcw_c_index(y_train, y_test, risk_scores):
    """
    计算基于逆删失概率加权的 Uno's IPCW C-index
    
    Parameters
    ----------
    y_train : np.array
        训练集生存标签 (用于估计删失概率分布)
    y_test : np.array
        测试集生存标签
    risk_scores : np.array
        模型预测的测试集风险分数
        
    Returns
    -------
    float
        Uno's IPCW C-index
    """
    y_train_surv = _to_sksurv_format(y_train)
    y_test_surv = _to_sksurv_format(y_test)
    
    # 注意：如果 test 集中的生存时间超出了 train 集的生存时间范围，IPCW计算会报错。
    # sksurv 内部使用 Kaplan-Meier 估计删失概率，要求 test 的最大时间 <= train 的最大观测时间。
    # tau 默认为 None，意味着评估覆盖所有可行区间。
    c_index_ipcw = concordance_index_ipcw(
        y_train_surv, 
        y_test_surv, 
        np.squeeze(risk_scores)
    )[0]
    
    return c_index_ipcw

def eval_time_dependent_auc(y_train, y_test, risk_scores, times=[90.0, 180.0, 365.0]):
    y_train_surv = _to_sksurv_format(y_train)
    y_test_surv = _to_sksurv_format(y_test)
    
    # 确定测试集内发生事件的有效时间区间
    # sksurv 要求评估时间必须 > 测试集最小事件时间 且 < 测试集最大观察时间
    mask_event = y_test_surv["event"]
    if not np.any(mask_event):
        return {"mean_AUC": np.nan}
        
    min_event_time = y_test_surv["time"][mask_event].min()
    max_test_time = y_test_surv["time"].max()
    
    valid_times = []
    time_name_mapping = {}
    for t in times:
        if min_event_time < t < max_test_time:
            valid_times.append(t)
            time_name_mapping[t] = f"{round(t/30.0)}Months"
            
    if not valid_times:
        # 如果没有一个时间点合法，直接返回空结果而不是抛出错误
        return {"mean_AUC": np.nan, "info": "No valid time points for AUC"}
        
    valid_times = np.array(valid_times)
    auc, mean_auc = cumulative_dynamic_auc(y_train_surv, y_test_surv, np.squeeze(risk_scores), valid_times)
    
    result = {f"AUC_{time_name_mapping[t]}": auc_val for t, auc_val in zip(valid_times, auc)}
    result["mean_AUC"] = mean_auc
    return result

def eval_brier_score_and_ibs(y_train, y_test, BSF, risk_scores, times=[90.0, 180.0, 365.0]):
    """
    计算特定时间点的 Brier Score 以及整体的 IBS (Integrated Brier Score)
    
    Parameters
    ----------
    y_train : np.array
        训练集生存标签
    y_test : np.array
        测试集生存标签
    BSF : DataFrame
        基于训练集计算得到的基线生存函数 (用于推断生存概率)
    risk_scores : np.array
        模型在测试集上预测的风险分数 (Hazard Ratio)
    times : list or np.array
        需要评估特定 Brier Score 的时间点(天数)。默认[90.0, 180.0, 365.0]
        
    Returns
    -------
    dict
        包含各指定时间点的 Brier Score 以及整体 IBS
    """
    y_train_surv = _to_sksurv_format(y_train)
    y_test_surv = _to_sksurv_format(y_test)
    
    # 确定测试集允许评估的时间边界
    min_time = y_test_surv["time"][y_test_surv["event"]].min()
    max_time = y_test_surv["time"].max()
    
    result = {}
    valid_times =[]
    time_name_mapping = {}
    
    # 1. 过滤指定的时间点，用于计算特定时间的 Brier Score
    for t in times:
        if min_time <= t <= max_time:
            valid_times.append(t)
            months = round(t / 30.0)
            time_name_mapping[t] = f"{months}Months"
            
    if valid_times:
        valid_times = np.array(valid_times)
        # 获取测试集在这些时间点的生存概率矩阵: Shape (n_samples, n_times)
        surv_prob_matrix = predict_surv_prob(BSF, risk_scores, valid_times)
        
        # 计算 Brier Score
        # sksurv 的 brier_score 返回两个数组：评估的时间点, 以及对应的 brier score
        _, bs_values = brier_score(y_train_surv, y_test_surv, surv_prob_matrix, valid_times)
        
        for t, bs_val in zip(valid_times, bs_values):
            result[f"BrierScore_{time_name_mapping[t]}"] = bs_val

    # 2. 计算高精度整体 IBS (Integrated Brier Score)
    # 选取测试集中所有有效的“事件发生时间”作为一个密集的网格来进行积分
    mask = (y_test_surv["time"] >= min_time) & (y_test_surv["time"] < max_time)
    dense_times = np.unique(y_test_surv["time"][mask])
    
    # 为了防止时间点过少导致积分无意义，加一个判断
    if len(dense_times) > 2:
        # 获取密集时间点的生存概率矩阵
        dense_surv_prob_matrix = predict_surv_prob(BSF, risk_scores, dense_times)
        # 计算 IBS
        ibs_val = integrated_brier_score(y_train_surv, y_test_surv, dense_surv_prob_matrix, dense_times)
        result["IBS_Overall"] = ibs_val
    else:
        warnings.warn("测试集有效时间点过少，无法计算整体 IBS。")
        result["IBS_Overall"] = np.nan
        
    return result

def _km_and_variance_at_t(event, time, target_t):
    """
    内部方法：手动计算特定时间点 target_t 的 Kaplan-Meier 生存率以及 Greenwood 方差。
    这是实现 GND 检验的底层核心。
    """
    idx = np.argsort(time)
    t = time[idx]
    e = event[idx]

    unique_t, counts = np.unique(t, return_counts=True)
    
    # 统计每个唯一时间点的事件数
    events_at_t = np.zeros_like(unique_t, dtype=float)
    for i, ut in enumerate(unique_t):
        events_at_t[i] = np.sum(e[t == ut])

    # 统计每个时间点的风险人数 (Number at risk)
    n_risk = np.zeros_like(unique_t, dtype=float)
    total = len(t)
    for i in range(len(unique_t)):
        n_risk[i] = total
        total -= counts[i]

    # 筛选出小于等于目标评估时间 target_t 的数据点
    mask = unique_t <= target_t
    if not np.any(mask):
        return 1.0, 0.0  # 如果在目标时间前没有任何事件，生存率为1，方差为0

    t_mask = unique_t[mask]
    e_mask = events_at_t[mask]
    n_mask = n_risk[mask]

    # 1. 计算 KM 生存率 S(t)
    p_survival = np.prod(1.0 - e_mask / n_mask)

    # 2. 计算 Greenwood 方差 V(S(t)) = S(t)^2 * sum (d_i / (n_i * (n_i - d_i)))
    var_terms = np.zeros_like(e_mask)
    for i in range(len(e_mask)):
        denom = n_mask[i] * (n_mask[i] - e_mask[i])
        if denom > 0:
            var_terms[i] = e_mask[i] / denom

    var_survival = (p_survival ** 2) * np.sum(var_terms)
    
    return p_survival, var_survival

def eval_calibration_and_gnd(y_test, BSF, risk_scores, times=[120.0, 180.0, 365.0], n_bins=10):
    y_test_surv = _to_sksurv_format(y_test)
    time_array = y_test_surv["time"]
    event_array = y_test_surv["event"]
    
    # 获取测试集的有效时间边界
    min_event_time = time_array[event_array].min() if np.any(event_array) else 0
    max_test_time = time_array.max()

    result = {}
    for t in times:
        # 增加前置检查：如果时间点 t 不在测试集有效范围内，直接跳过该时间点
        if not (min_event_time < t < max_test_time):
            continue

        time_key = f"{round(t/30.0)}Months"
        pred_probs = predict_surv_prob(BSF, risk_scores, [t]).flatten()
        
        try:
            groups = pd.qcut(pred_probs, q=n_bins, labels=False, duplicates='drop')
            unique_groups = np.unique(groups)
        except (ValueError, IndexError):
            continue
            
        expected_probs, observed_probs = [], []
        gnd_chi2 = 0.0
        
        for g in unique_groups:
            idx_g = (groups == g)
            # 防御性检查：确保组内有样本
            if not np.any(idx_g):
                continue
            
            expected_p = np.mean(pred_probs[idx_g])
            obs_p, obs_var = _km_and_variance_at_t(event_array[idx_g], time_array[idx_g], t)
            
            expected_probs.append(float(expected_p))
            observed_probs.append(float(obs_p))
            
            if obs_var > 1e-9:
                gnd_chi2 += ((obs_p - expected_p) ** 2) / obs_var
        
        df = len(expected_probs) - 1
        p_value = chi2.sf(gnd_chi2, df) if df > 0 else np.nan
        
        result[f"Calibration_Plot_{time_key}"] = {"Expected": expected_probs, "Observed": observed_probs}
        result[f"GND_pvalue_{time_key}"] = float(p_value)

    return result

def eval_dca(y_test, BSF, risk_scores, times=[90.0, 180.0, 365.0], thresholds=np.arange(0.01, 1.00, 0.01)):
    """
    计算特定时间点的生存 DCA (Decision Curve Analysis) 曲线数据
    
    Parameters
    ----------
    y_test : np.array
        测试集生存标签
    BSF : DataFrame
        基线生存函数 (用于计算每个个体的生存概率)
    risk_scores : np.array
        测试集预测风险分数
    times : list
        评估时间点(天数)，默认 [90, 180, 365]
    thresholds : np.array
        医生愿意采取干预措施的风险概率阈值数组，默认从 1% 到 99%
        
    Returns
    -------
    dict
        包含各时间点 DCA 数据的字典，供直接画图使用
    """
    y_test_surv = _to_sksurv_format(y_test)
    time_array = y_test_surv["time"]
    event_array = y_test_surv["event"]
    
    total_N = len(time_array)
    result = {}
    
    for t in times:
        months = round(t / 30.0)
        time_key = f"{months}Months"
        
        # 1. 预测的“事件发生概率” = 1 - 生存概率
        pred_surv_probs = predict_surv_prob(BSF, risk_scores, [t]).flatten()
        pred_event_probs = 1.0 - pred_surv_probs
        
        # 2. 计算基准线 (Treat All): 假设所有人都是高风险
        # 对全人群跑 KM，计算时间 t 的真实生存率
        s_all_t, _ = _km_and_variance_at_t(event_array, time_array, t)
        e_all_t = 1.0 - s_all_t  # 真实的全人群事件发生率
        
        nb_model_list = []
        nb_all_list =[]
        nb_none_list =[]
        
        for pt in thresholds:
            # 防止除以 0 的极端情况
            if pt >= 1.0:
                continue
                
            odds = pt / (1.0 - pt)
            
            # --- 计算 Treat All 的净收益 ---
            # 所有人干预的真阳性率即为全人群事件率，假阳性率即为全人群未发生事件率
            nb_all = e_all_t - s_all_t * odds
            nb_all_list.append(nb_all)
            
            # --- 计算 Treat None 的净收益 ---
            nb_none_list.append(0.0)
            
            # --- 计算 Model 的净收益 ---
            # 筛选出被模型判定为高风险的患者 (预测事件率 >= 阈值 pt)
            idx_pos = (pred_event_probs >= pt)
            n_pos = np.sum(idx_pos)
            
            if n_pos == 0:
                nb_model_list.append(0.0)
            else:
                # 对这部分被模型判定为阳性的群体，跑 KM 算真实事件率
                s_pos_t, _ = _km_and_variance_at_t(event_array[idx_pos], time_array[idx_pos], t)
                e_pos_t = 1.0 - s_pos_t
                
                # 生存分析 DCA 公式: (N_pos / N) *[E_pos - S_pos * (pt / 1-pt)]
                tp_rate = (n_pos / total_N) * e_pos_t
                fp_rate = (n_pos / total_N) * s_pos_t
                nb_model = tp_rate - fp_rate * odds
                nb_model_list.append(nb_model)
                
        result[f"DCA_{time_key}"] = {
            "Thresholds": thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds,
            "NetBenefit_Model": nb_model_list,
            "NetBenefit_All": nb_all_list,
            "NetBenefit_None": nb_none_list
        }
        
    return result