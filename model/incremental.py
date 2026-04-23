import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
import warnings


from .utils import (
    _check_config,
    _prepare_surv_data,
    baseline_survival_function,
    predict_surv_prob,
    eval_c_index,
    eval_ipcw_c_index,
    eval_time_dependent_auc,
    eval_brier_score_and_ibs,
    eval_calibration_and_gnd,
    eval_dca
)

class IncrementalAutoCox(nn.Module):
    """
    Incremental Cox network implemented in PyTorch.
    Features separate branches for old/new features to prevent catastrophic forgetting.
    """

    def __init__(self, hidden_layers_nodes, encoder_config=None, config=None):
        super(IncrementalAutoCox, self).__init__()

        self.encoder_config = {
            'dropout': 0.0,
            'loss_alpha': 0.5,
            'loss_alpha2': 0.5,
            'warm_up_epoch': 0,
            'L2_reg': 0.0,
            'n_hidden_1': 64,
            'n_hidden_2': 32,
            'n_input': None,
            'feature_split': None,
            'old_lr_ratio': 0.0,  # 冻结或微调旧特征
            'new_lr_ratio': 1.0,
        }
        if encoder_config:
            self.encoder_config.update(encoder_config)

        self.config = config or {}
        _check_config(self.config)

        self.encoder_config.setdefault('loss_alpha2', self.encoder_config['loss_alpha'])
        if self.encoder_config['n_input'] is None:
            raise ValueError('`n_input` must be specified in encoder config.')

        self.hidden_layers_nodes = hidden_layers_nodes
        assert self.hidden_layers_nodes[-1] == 1, 'Last output dimension must be 1.'

        torch.manual_seed(int(self.config['seed']))
        
        # 构建网络结构
        self._build_network()
        self.BSF = None  # 基线生存函数对象

    def _build_network(self):
        n_input = self.encoder_config['n_input']
        n_hidden_1 = self.encoder_config['n_hidden_1']
        n_hidden_2 = self.encoder_config['n_hidden_2']
        feature_split = self.encoder_config.get('feature_split')
        dropout_rate = self.encoder_config['dropout']

        # === 1. Encoder 编码器构建 ===
        if feature_split is not None and len(feature_split) == 2:
            self.old_dim, self.new_dim = feature_split
            if self.old_dim + self.new_dim != n_input:
                raise ValueError('feature_split 维度和必须等于 n_input.')
            
            # 分支编码器 (无激活函数，后续合并后加 ReLU)
            self.encoder_old_dense = nn.Linear(self.old_dim, n_hidden_1, bias=False)
            self.encoder_new_dense = nn.Linear(self.new_dim, n_hidden_1, bias=False)
            # 合并偏置项
            self.encoder_merge_bias = nn.Parameter(torch.zeros(n_hidden_1))
        else:
            self.encoder_dense_1 = nn.Linear(n_input, n_hidden_1)

        # 共享特征表达层
        self.encoder_act_1 = nn.ReLU()
        self.encoder_dropout_1 = nn.Dropout(p=dropout_rate)
        self.encoder_dense_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.encoder_act_2 = nn.ReLU()
        self.encoder_dropout_2 = nn.Dropout(p=dropout_rate)

        # === 2. Decoder 解码器构建 ===
        if feature_split is not None and len(feature_split) == 2:
            self.decoder_old = nn.Linear(n_hidden_2, self.old_dim)
            self.decoder_new = nn.Linear(n_hidden_2, self.new_dim)
        else:
            self.decoder_dense = nn.Linear(n_hidden_2, n_input)

        # === 3. 生存风险预测头 (Risk Prediction Head) ===
        layers =[]
        input_dim = n_hidden_2
        for num_nodes in self.hidden_layers_nodes:
            layers.append(nn.Linear(input_dim, num_nodes))
            # 最后一层输出 1 维对数风险比，不要激活函数
            if num_nodes != 1:
                layers.append(nn.ReLU())
            input_dim = num_nodes
            
        self.risk_head = nn.Sequential(*layers)

    def forward(self, inputs):
        feature_split = self.encoder_config.get('feature_split')
        
        # --- 编码过程 ---
        if feature_split is not None and len(feature_split) == 2:
            x_old = inputs[:, :self.old_dim]
            x_new = inputs[:, self.old_dim:]
            # 融合分支
            out_old = self.encoder_old_dense(x_old)
            out_new = self.encoder_new_dense(x_new)
            x = self.encoder_act_1(out_old + out_new + self.encoder_merge_bias)
        else:
            x = self.encoder_act_1(self.encoder_dense_1(inputs))

        x = self.encoder_dropout_1(x)
        x = self.encoder_act_2(self.encoder_dense_2(x))
        bottleneck = self.encoder_dropout_2(x)

        # --- 解码重构过程 ---
        if feature_split is not None and len(feature_split) == 2:
            dec_old = self.decoder_old(bottleneck)
            dec_new = self.decoder_new(bottleneck)
            decoder_pred = torch.cat([dec_old, dec_new], dim=-1)
        else:
            decoder_pred = self.decoder_dense(bottleneck)

        # --- 风险预测 ---
        y_hat = self.risk_head(bottleneck)

        return y_hat, decoder_pred
    
    def _cox_loss(self, data_X, y_true, y_hat, decoder_pred, loss_alpha):
            """
            纯 PyTorch 矩阵化实现的 Cox + 重构联合损失函数
            """
            # 确保维度为 1D 向量 (N,)
            y_true = y_true.view(-1)
            y_hat = y_hat.view(-1)
            
            # y_true 的绝对值为生存时间，正负号判断是否为事件
            y_label_t = torch.abs(y_true)
            y_label_e = (y_true > 0).float()
            obs = torch.sum(y_label_e) + 1e-8
            
            # === 1. Cox Breslow 偏似然损失 ===
            # 利用广播机制构建风险矩阵 (Risk Matrix)
            # risk_matrix[i, j] = 1 表示患者 j 存活时间 >= 患者 i，即 j 属于 i 的风险集
            risk_matrix = (y_label_t.unsqueeze(1) <= y_label_t.unsqueeze(0)).float()
            
            y_hat_hr = torch.exp(y_hat)
            # 矩阵乘法：并行计算所有事件患者的风险集对数和
            risk_set_sum = torch.matmul(risk_matrix, y_hat_hr)
            log_risk_set = torch.log(risk_set_sum + 1e-12)
            
            # Breslow 损失公式: sum_i ( E_i *[log(sum_{j \in R(t_i)} exp(y_j)) - y_i] ) / N_obs
            breslow_loss = torch.sum(y_label_e * (log_risk_set - y_hat)) / obs
            
            # === 2. L2 正则化惩罚 ===
            # 模拟 TF 中的 kernel_regularizer
            l2_scale = self.encoder_config.get('L2_reg', 0.0)
            reg_loss = 0.0
            if l2_scale > 0:
                for name, param in self.named_parameters():
                    # 仅对线性层的权重(weight)做正则，忽略偏置(bias)
                    if 'weight' in name:
                        reg_loss += torch.sum(param ** 2)
                reg_loss = l2_scale * reg_loss
                
            # === 3. 自编码器重构误差 (MSE) ===
            reconstruction_error = torch.mean((decoder_pred - data_X) ** 2)
            
            # === 4. 联合损失加权 ===
            total_loss = loss_alpha * (breslow_loss + reg_loss) + (1.0 - loss_alpha) * reconstruction_error
            
            return total_loss

    def _create_optimizer(self):
            """
            PyTorch 风格的参数组优化器构建。
            为旧特征分支、新特征分支和共享层分配不同的学习率系数。
            """
            base_lr = self.config.get('learning_rate', 0.001)
            optimizer_name = self.config.get('optimizer', 'adam').lower()
            
            feature_split = self.encoder_config.get('feature_split')
            old_ratio = self.encoder_config.get('old_lr_ratio', 1.0)
            new_ratio = self.encoder_config.get('new_lr_ratio', 1.0)

            # 如果没有拆分特征，使用单一学习率的常规模式
            if feature_split is None:
                params = self.parameters()
            else:
                # --- 关键重构：按功能将参数归类 ---
                old_params = []
                new_params = []
                shared_params = []

                # 提取旧分支参数
                old_params += list(self.encoder_old_dense.parameters())
                old_params += list(self.decoder_old.parameters())
                
                # 提取新分支参数
                new_params += list(self.encoder_new_dense.parameters())
                new_params += list(self.decoder_new.parameters())
                
                # 提取其余共享层参数 (包括偏置项、Bottleneck层、Risk Head)
                # 通过排除法获取
                all_special_params_ids = set(id(p) for p in old_params + new_params)
                shared_params = [p for p in self.parameters() if id(p) not in all_special_params_ids]

                # 构造 PyTorch 参数组
                params = [
                    {'params': old_params, 'lr': base_lr * old_ratio, 'name': 'old_branch'},
                    {'params': new_params, 'lr': base_lr * new_ratio, 'name': 'new_branch'},
                    {'params': shared_params, 'lr': base_lr, 'name': 'shared_layers'}
                ]

            # 实例化具体优化器
            if optimizer_name == 'sgd':
                return optim.SGD(params, lr=base_lr)
            elif optimizer_name == 'rms':
                return optim.RMSprop(params, lr=base_lr)
            else:
                return optim.Adam(params, lr=base_lr)

    def predict(self, X, output_margin=True):
        """
        风险预测接口
        """
        self.eval()  # 必须：切换到评估模式 (关闭 Dropout)
        with torch.no_grad():
            # 支持传入 numpy 或 polars.Series/DataFrame，统一转为 tensor
            if isinstance(X, (pl.DataFrame, pl.Series)):
                X = X.to_numpy()
            
            X_tensor = torch.as_tensor(X, dtype=torch.float32)
            log_hr, _ = self(X_tensor)
            
            if output_margin:
                return log_hr.cpu().numpy()
            return torch.exp(log_hr).cpu().numpy()

    def predict_survival_function(self, X, times=[90, 180, 365]):
        """
        基于 BSF 预测个体在特定时间点的生存函数，返回 Polars DataFrame
        """
        if self.BSF is None:
            raise RuntimeError('Baseline survival function has not been computed. Call `train` first.')
            
        pred_hr = self.predict(X, output_margin=False)
        # 调用 utils.py 中的推断函数 (结果为 numpy 矩阵)
        survf_matrix = predict_surv_prob(self.BSF, pred_hr, times)
        
        # 将 numpy 矩阵转换为 Polars DataFrame
        # 列名映射为: Time_90, Time_180 ...
        column_names = [f"Time_{t}" for t in times]
        return pl.from_numpy(survf_matrix, schema=column_names)

    def evals(self, data_X, data_y):
        """
        快捷评估接口 (C-index)
        """
        preds = self.predict(data_X, output_margin=True)
        return eval_c_index(data_y, preds)
    
    def fit(self,
            data_X,
            data_y,
            test_x=None,
            test_y=None,
            num_steps=1000,
            num_skip_steps=100,
            save_model_path='',
            history_path='training_history.json', # 新增参数
            silent=False):
        
        # --- 1. 数据转换 ---
        if isinstance(data_X, (pl.DataFrame, pl.Series)): data_X = data_X.to_numpy()
        if isinstance(test_x, (pl.DataFrame, pl.Series)) and test_x is not None: test_x = test_x.to_numpy()
        
        data_X_eff = torch.as_tensor(data_X, dtype=torch.float32)
        data_y_eff = torch.as_tensor(np.asarray(data_y), dtype=torch.float32)
        
        optimizer = self._create_optimizer()
        
        # 用于保存训练过程中的指标
        history = {
            "steps": [],
            "train_c_index": [],
            "test_c_index": [],
            "final_clinical_metrics": {}
        }
        
        # --- 2. 训练迭代 ---
        for step in range(num_steps + 1):
            self.train() # 调用 nn.Module.train() 切换模式
            optimizer.zero_grad()
            
            current_alpha = self.encoder_config['loss_alpha'] if step < self.encoder_config['warm_up_epoch'] \
                            else self.encoder_config['loss_alpha2']
            
            y_hat, decoder_pred = self(data_X_eff)
            loss = self._cox_loss(data_X_eff, data_y_eff, y_hat, decoder_pred, current_alpha)
            
            loss.backward()
            optimizer.step()
            
            # 定期记录过程指标
            if step % num_skip_steps == 0:
                train_c = self.evals(data_X, data_y)
                history["steps"].append(int(step))
                history["train_c_index"].append(float(train_c))
                
                msg = f"Step: {step} | Loss: {loss.item():.4f} | Train C: {train_c:.4f}"
                
                if test_x is not None and test_y is not None:
                    test_c = self.evals(test_x, test_y)
                    history["test_c_index"].append(float(test_c))
                    msg += f" | Test C: {test_c:.4f}"
                
                if not silent: print(msg)

        # --- 3. 计算 BSF (后续指标的基础) ---
        self.eval()
        with torch.no_grad():
            final_hr = self.predict(data_X, output_margin=False)
            self.BSF = baseline_survival_function(data_y, final_hr)

        # --- 4. 深度评估并保存到 history ---
        if test_x is not None and test_y is not None:
            test_hr = self.predict(test_x, output_margin=False)
            
            # (1) IPCW C-index
            history["final_clinical_metrics"]["IPCW_C_Index"] = float(eval_ipcw_c_index(data_y, test_y, test_hr))
            
            # (2) Time-dependent AUC
            auc_res = eval_time_dependent_auc(data_y, test_y, test_hr, times=[90, 180, 365])
            history["final_clinical_metrics"].update({k: float(v) for k, v in auc_res.items()})
            
            # (3) Brier Score & IBS
            ibs_res = eval_brier_score_and_ibs(data_y, test_y, self.BSF, test_hr, times=[90, 180, 365])
            history["final_clinical_metrics"].update({k: float(v) for k, v in ibs_res.items()})
            
            # (4) Calibration & GND (可选：如果想保存画图数据)
            cal_res = eval_calibration_and_gnd(test_y, self.BSF, test_hr, times=[90, 180, 365])
            # 注意：Calibration 的数据是列表，不需要转 float
            history["final_clinical_metrics"].update(cal_res)

             # 【新增：(5) DCA 决策曲线数据】
            dca_res = eval_dca(test_y, self.BSF, test_hr, times=[90, 180, 365])
            history["final_clinical_metrics"].update(dca_res)

        # --- 5. 写入 JSON 文件 ---
        if history_path:
            import json
            # 自定义序列化处理，确保所有 numpy 类型都能转为 Python 原生类型
            def default_converter(obj):
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4, default=default_converter)
            if not silent: print(f"所有评估指标已保存至: {history_path}")

        # --- 6. 模型保存 ---
        if save_model_path:
            torch.save(self.state_dict(), save_model_path)

        return history

    def load_model(self, model_path):
        """加载模型权重"""
        self.load_state_dict(torch.load(model_path))
        self.eval()