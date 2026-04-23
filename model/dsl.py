import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
import json


from .utils import (
    _check_config,
    baseline_survival_function,
    predict_surv_prob,
    eval_c_index,
    eval_ipcw_c_index,
    eval_time_dependent_auc,
    eval_brier_score_and_ibs,
    eval_calibration_and_gnd,
    eval_dca
)

class AutoCox(nn.Module):
    """
    Standard Deep Cox network with an Autoencoder implemented in PyTorch.
    (无增量特征分支的基础版本)
    """

    def __init__(self, hidden_layers_nodes, encoder_config=None, config=None):
        super(AutoCox, self).__init__()

        self.encoder_config = {
            'dropout': 0.0,
            'loss_alpha': 0.5,
            'loss_alpha2': 0.5,
            'warm_up_epoch': 0,
            'L2_reg': 0.0,
            'n_hidden_1': 64,
            'n_hidden_2': 32,
            'n_input': None,
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

        # 设置随机种子
        seed = int(self.config.get('seed', 42))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self._build_network()
        self.BSF = None

    def _build_network(self):
        n_input = self.encoder_config['n_input']
        n_hidden_1 = self.encoder_config['n_hidden_1']
        n_hidden_2 = self.encoder_config['n_hidden_2']
        dropout_rate = self.encoder_config['dropout']

        # === 1. 单路 Encoder 编码器 ===
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # === 2. Decoder 解码器 ===
        self.decoder = nn.Linear(n_hidden_2, n_input)

        # === 3. 生存风险预测头 (Risk Prediction Head) ===
        layers = []
        input_dim = n_hidden_2
        for num_nodes in self.hidden_layers_nodes:
            layers.append(nn.Linear(input_dim, num_nodes))
            if num_nodes != 1:
                layers.append(nn.ReLU())
            input_dim = num_nodes
            
        self.risk_head = nn.Sequential(*layers)

    def forward(self, inputs):
        # 提取隐层特征 (Bottleneck)
        bottleneck = self.encoder(inputs)
        
        # 重构输入
        decoder_pred = self.decoder(bottleneck)
        
        # 预测风险分数 (Log Hazard Ratio)
        y_hat = self.risk_head(bottleneck)

        return y_hat, decoder_pred

    def _create_optimizer(self):
        base_lr = self.config.get('learning_rate', 0.001)
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        # 标准模型：所有参数使用相同的学习率
        if optimizer_name == 'sgd':
            return optim.SGD(self.parameters(), lr=base_lr)
        elif optimizer_name == 'rms':
            return optim.RMSprop(self.parameters(), lr=base_lr)
        else:
            return optim.Adam(self.parameters(), lr=base_lr)

    def _cox_loss(self, data_X, y_true, y_hat, decoder_pred, loss_alpha):
        y_true = y_true.view(-1)
        y_hat = y_hat.view(-1)
        
        y_label_t = torch.abs(y_true)
        y_label_e = (y_true > 0).float()
        obs = torch.sum(y_label_e) + 1e-8
        
        # 纯张量矩阵化计算 Breslow 偏似然
        risk_matrix = (y_label_t.unsqueeze(1) <= y_label_t.unsqueeze(0)).float()
        y_hat_hr = torch.exp(y_hat)
        risk_set_sum = torch.matmul(risk_matrix, y_hat_hr)
        log_risk_set = torch.log(risk_set_sum + 1e-12)
        
        breslow_loss = torch.sum(y_label_e * (log_risk_set - y_hat)) / obs
        
        # L2 正则化
        l2_scale = self.encoder_config.get('L2_reg', 0.0)
        reg_loss = 0.0
        if l2_scale > 0:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.sum(param ** 2)
            reg_loss = l2_scale * reg_loss
            
        # MSE 重构误差
        reconstruction_error = torch.mean((decoder_pred - data_X) ** 2)
        
        # 加权融合
        total_loss = loss_alpha * (breslow_loss + reg_loss) + (1.0 - loss_alpha) * reconstruction_error
        return total_loss

    def predict(self, X, output_margin=True):
        self.eval()
        with torch.no_grad():
            if isinstance(X, (pl.DataFrame, pl.Series)):
                X = X.to_numpy()
            X_tensor = torch.as_tensor(X, dtype=torch.float32)
            log_hr, _ = self(X_tensor)
            
            if output_margin:
                return log_hr.cpu().numpy()
            return torch.exp(log_hr).cpu().numpy()

    def evals(self, data_X, data_y):
        preds = self.predict(data_X, output_margin=True)
        return eval_c_index(data_y, preds)

    def predict_survival_function(self, X, times=[90, 180, 365]):
        if self.BSF is None:
            raise RuntimeError('Baseline survival function has not been computed.')
        pred_hr = self.predict(X, output_margin=False)
        survf_matrix = predict_surv_prob(self.BSF, pred_hr, times)
        column_names = [f"Time_{t}" for t in times]
        return pl.from_numpy(survf_matrix, schema=column_names)

    def fit(self,
            data_X,
            data_y,
            test_x=None,
            test_y=None,
            num_steps=1000,
            num_skip_steps=100,
            save_model_path='',
            history_path='training_history_autocox.json',
            silent=False):
        
        if isinstance(data_X, (pl.DataFrame, pl.Series)): data_X = data_X.to_numpy()
        if isinstance(test_x, (pl.DataFrame, pl.Series)) and test_x is not None: test_x = test_x.to_numpy()
        
        data_X_eff = torch.as_tensor(data_X, dtype=torch.float32)
        data_y_eff = torch.as_tensor(np.asarray(data_y), dtype=torch.float32)
        
        optimizer = self._create_optimizer()
        
        history = {"steps": [], "train_c_index": [], "test_c_index": [], "final_clinical_metrics": {}}
        
        for step in range(num_steps + 1):
            self.train() 
            optimizer.zero_grad()
            
            current_alpha = self.encoder_config['loss_alpha'] if step < self.encoder_config['warm_up_epoch'] \
                            else self.encoder_config['loss_alpha2']
            
            y_hat, decoder_pred = self(data_X_eff)
            loss = self._cox_loss(data_X_eff, data_y_eff, y_hat, decoder_pred, current_alpha)
            
            loss.backward()
            optimizer.step()
            
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

        # 训练结束，计算 BSF 并进行深度评估
        self.eval()
        with torch.no_grad():
            final_hr = self.predict(data_X, output_margin=False)
            self.BSF = baseline_survival_function(data_y, final_hr)

        if test_x is not None and test_y is not None:
            test_hr = self.predict(test_x, output_margin=False)
            history["final_clinical_metrics"]["IPCW_C_Index"] = float(eval_ipcw_c_index(data_y, test_y, test_hr))
            history["final_clinical_metrics"].update({k: float(v) for k, v in eval_time_dependent_auc(data_y, test_y, test_hr, times=[90, 180, 365]).items()})
            history["final_clinical_metrics"].update({k: float(v) for k, v in eval_brier_score_and_ibs(data_y, test_y, self.BSF, test_hr, times=[90, 180, 365]).items()})
            history["final_clinical_metrics"].update(eval_calibration_and_gnd(test_y, self.BSF, test_hr, times=[90, 180, 365]))
            history["final_clinical_metrics"].update(eval_dca(test_y, self.BSF, test_hr, times=[90, 180, 365]))

        if history_path:
            def default_converter(obj):
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                raise TypeError(f"Not serializable: {type(obj)}")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4, default=default_converter)

        if save_model_path:
            torch.save(self.state_dict(), save_model_path)

        return history

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()