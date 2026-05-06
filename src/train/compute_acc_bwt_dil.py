import os
import sys
import itertools
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ====================== 配置区 ======================
H5_BASE_DIR = ""

DOMAIN_H5_FILES = {
    "activitynet": os.path.join(H5_BASE_DIR, "act_vit.h5"),
    "ucf101":      os.path.join(H5_BASE_DIR, "ucf_vit_25.h5"),
    "fcvid":       os.path.join(H5_BASE_DIR, "fcvid_vit_train.h5")
}

SPLIT_BASE = "/domain_incremental"

DOMAIN_SPLIT_DIRS = {
    "activitynet": os.path.join(SPLIT_BASE, "Activitynet"),
    "ucf101": os.path.join(SPLIT_BASE, "ucf101"),
    "fcvid": os.path.join(SPLIT_BASE, "fcvid")
}

DOMAIN_FILE_NAMES = {
    "activitynet": {"train": "train.txt", "val": "val.txt"},
    "ucf101":      {"train": "train.txt", "val": "val.txt"},
    "fcvid":       {"train": "fcv_train.txt", "val": "fcv_val.txt"}
}

DOMAIN_DATASET_TYPES = {
    "activitynet": "actnet",
    "ucf101": "ucf",
    "fcvid": "fcvid"
}

# ========== Transformer 参数 ==========
D_MODEL = 768
NHEAD = 8
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
POOLING_MODE = 'cls'

# ========== 训练参数 ==========
TRAIN_EPOCHS = 20_15_100  # 此常量仅用于日志文件名，实际训练 epoch 在 update 中动态确定
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 正则化 / 回放超参数 ==========
EWC_LAMBDA = 1000.0
SI_LAMBDA = 1.0
DERPP_ALPHA = 0.5
MEMORY_SIZE = 2000

# ====================== 数据加载函数 ======================
def load_split_file(txt_path, dataset_type):
    video_names = []
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if dataset_type == 'actnet':
                if len(parts) >= 3:
                    video_names.append(parts[0])
                    labels.append(int(parts[2]))
            else:
                if len(parts) >= 2:
                    video_names.append(parts[0])
                    labels.append(int(parts[1]))
    return video_names, np.array(labels)

def load_features_from_h5(h5_path, video_names, aggregate=False):
    features = []
    with h5py.File(h5_path, 'r') as f:
        for name in tqdm(video_names, desc=f"Loading {os.path.basename(h5_path)}", leave=False):
            grp = f.get(name)
            if grp is None:
                base = os.path.basename(name)
                base_no_ext = os.path.splitext(base)[0]
                grp = f.get(base_no_ext) or f.get(base)
            if grp is None:
                print(f"Warning: Video {name} not found. Using zero vector.")
                vec = np.zeros((1, D_MODEL), dtype=np.float32)
            else:
                vec = grp['vectors'][:]
            if aggregate:
                if vec.ndim == 2:
                    vec = vec.mean(axis=0)
            features.append(vec.astype(np.float32))
    return features

def load_domain_data(domain_name, phase='train'):
    split_dir = DOMAIN_SPLIT_DIRS[domain_name]
    file_name = DOMAIN_FILE_NAMES[domain_name][phase]
    txt_file = os.path.join(split_dir, file_name)

    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Missing file: {txt_file}")
    dataset_type = DOMAIN_DATASET_TYPES[domain_name]
    video_names, labels = load_split_file(txt_file, dataset_type)
    h5_file = DOMAIN_H5_FILES[domain_name]
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Missing H5 file: {h5_file}")
    features = load_features_from_h5(h5_file, video_names, aggregate=False)
    return features, labels

# ====================== 自定义 Dataset ======================
class SequenceDataset(Dataset):
    def __init__(self, features_list, labels, is_replay=None):
        self.features = features_list
        self.labels = labels
        self.is_replay = is_replay if is_replay is not None else [False] * len(features_list)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        is_replay = torch.tensor(self.is_replay[idx], dtype=torch.bool)
        return feat, label, is_replay

def collate_fn(batch):
    features, labels, is_replay = zip(*batch)
    padded_feats = pad_sequence(features, batch_first=False)
    lengths = torch.tensor([f.size(0) for f in features])
    mask = torch.zeros(padded_feats.size(1), padded_feats.size(0), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    mask = mask.to(DEVICE)
    labels = torch.stack(labels).to(DEVICE)
    is_replay = torch.stack(is_replay).to(DEVICE)
    return padded_feats.to(DEVICE), mask, labels, is_replay

# ====================== Transformer 分类器模型 ======================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout, pooling='cls'):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if pooling == 'cls' else None
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask=None):
        src = self.input_proj(src)
        seq_len, batch_size, _ = src.shape
        src = src + self.pos_encoder[:, :seq_len, :].transpose(0, 1)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(-1, batch_size, -1)
            src = torch.cat([cls_tokens, src], dim=0)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        output = self.transformer(src, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == 'cls':
            video_repr = output[0]
        elif self.pooling == 'mean':
            if src_key_padding_mask is not None:
                lengths = (~src_key_padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
                output_permuted = output.permute(1, 0, 2)
                valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
                video_repr = (output_permuted * valid_mask).sum(dim=1) / lengths
            else:
                video_repr = output.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(video_repr)

# ====================== 增量学习分类器（支持多种方法） ======================
class TrainableTransformerClassifier:
    def __init__(self, input_dim, method, device=DEVICE):
        self.input_dim = input_dim
        self.method = method
        self.device = device
        self.classes = []                 # (domain, original_label)
        self.class_to_global_idx = {}     # (domain, original_label) -> global_index
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        # 回放缓冲区
        self.memory_x = []
        self.memory_y = []
        self.memory_logits = []
        self.memory_domains = []

        # Joint 全量数据存储
        self.joint_data_x = []
        self.joint_data_y = []

        # EWC 相关
        self.ewc_fisher = {}
        self.ewc_old_params = {}

        # SI 相关
        self.si_omega = {}
        self.si_old_params = {}
        self.si_prev_params = None
        self.si_accumulated_grad = None

        # DER++ 旧模型副本
        self.old_model = None

    def _expand_model(self, new_global_indices):
        old_num_classes = len(self.classes) - len(new_global_indices)
        new_num_classes = len(self.classes)

        new_model = TransformerClassifier(
            input_dim=self.input_dim,
            num_classes=new_num_classes,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            pooling=POOLING_MODE
        ).to(self.device)

        if self.model is not None:
            with torch.no_grad():
                # 拷贝分类头旧权重
                new_model.classifier.weight[:old_num_classes] = self.model.classifier.weight
                new_model.classifier.bias[:old_num_classes] = self.model.classifier.bias
                # 拷贝主干网络参数
                new_model.input_proj.load_state_dict(self.model.input_proj.state_dict())
                new_model.pos_encoder.data.copy_(self.model.pos_encoder.data)
                if self.model.cls_token is not None:
                    new_model.cls_token.data.copy_(self.model.cls_token.data)
                new_model.transformer.load_state_dict(self.model.transformer.state_dict())

        self.model = new_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # ---------- 扩展持续学习相关状态字典 (EWC / SI) ----------
        def _extend_classifier_state(state_dict, old_dim, new_dim):
            if state_dict is None:
                return
            for name in list(state_dict.keys()):
                if 'classifier' in name:
                    tensor = state_dict[name]
                    if tensor.dim() == 2:  # weight
                        new_tensor = torch.zeros(new_dim, tensor.size(1),
                                                 device=tensor.device, dtype=tensor.dtype)
                        new_tensor[:old_dim] = tensor
                        state_dict[name] = new_tensor
                    elif tensor.dim() == 1:  # bias
                        new_tensor = torch.zeros(new_dim, device=tensor.device, dtype=tensor.dtype)
                        new_tensor[:old_dim] = tensor
                        state_dict[name] = new_tensor

        if self.method == 'ewc':
            _extend_classifier_state(self.ewc_fisher, old_num_classes, new_num_classes)
            _extend_classifier_state(self.ewc_old_params, old_num_classes, new_num_classes)
        elif self.method == 'si':
            _extend_classifier_state(self.si_omega, old_num_classes, new_num_classes)
            _extend_classifier_state(self.si_old_params, old_num_classes, new_num_classes)

    def _update_memory(self, features_list, global_labels, logits_list=None):
        if self.method == 'joint':
            self.joint_data_x.extend(features_list)
            self.joint_data_y.extend(global_labels.tolist())
            return

        if self.method not in ['er', 'gdumb', 'derpp']:
            return

        for i, (feat, label) in enumerate(zip(features_list, global_labels)):
            self.memory_x.append(feat.copy())
            self.memory_y.append(label)
            if self.method == 'derpp' and logits_list is not None:
                self.memory_logits.append(logits_list[i])

        if len(self.memory_y) > MEMORY_SIZE:
            if self.method == 'gdumb':
                self._gdumb_balance()
            else:
                indices = np.random.choice(len(self.memory_y), MEMORY_SIZE, replace=False)
                self.memory_x = [self.memory_x[i] for i in indices]
                self.memory_y = [self.memory_y[i] for i in indices]
                if self.method == 'derpp':
                    self.memory_logits = [self.memory_logits[i] for i in indices]

    def _gdumb_balance(self):
        labels = np.array(self.memory_y)
        unique_labels = np.unique(labels)
        target_per_class = max(1, MEMORY_SIZE // len(unique_labels))
        keep_indices = []
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) > target_per_class:
                idx = np.random.choice(idx, target_per_class, replace=False)
            keep_indices.extend(idx)
        if len(keep_indices) > MEMORY_SIZE:
            keep_indices = np.random.choice(keep_indices, MEMORY_SIZE, replace=False)
        self.memory_x = [self.memory_x[i] for i in keep_indices]
        self.memory_y = [self.memory_y[i] for i in keep_indices]
        if self.method == 'derpp':
            self.memory_logits = [self.memory_logits[i] for i in keep_indices]

    def _compute_ewc_fisher(self, dataloader):
        self.model.eval()
        fisher = {name: torch.zeros_like(p) for name, p in self.model.named_parameters()}
        for src, mask, batch_y, _ in dataloader:
            self.model.zero_grad()
            logits = self.model(src, src_key_padding_mask=~mask)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.pow(2) / len(dataloader)
        self.ewc_fisher = fisher
        self.ewc_old_params = {name: p.clone().detach() for name, p in self.model.named_parameters()}

    def _update_si_omega(self):
        current_params = {name: p.clone().detach() for name, p in self.model.named_parameters()}
        if self.si_prev_params is not None:
            for name in current_params:
                delta = current_params[name] - self.si_old_params[name]
                omega = self.si_omega.get(name, torch.zeros_like(current_params[name]))
                if self.si_accumulated_grad is not None and name in self.si_accumulated_grad:
                    omega += delta * self.si_accumulated_grad[name]
                self.si_omega[name] = omega
        self.si_old_params = current_params
        self.si_prev_params = self.si_old_params

    def update(self, features_list, labels, domain_name):
        unique_labels = np.unique(labels)
        new_pairs = [(domain_name, lbl) for lbl in unique_labels
                     if (domain_name, lbl) not in self.class_to_global_idx]

        if len(new_pairs) == 0:
            print("  No new classes to add, skipping training.")
            return

        for pair in new_pairs:
            global_idx = len(self.classes)
            self.class_to_global_idx[pair] = global_idx
            self.classes.append(pair)

        self._expand_model([self.class_to_global_idx[p] for p in new_pairs])

        new_global_labels = np.array([self.class_to_global_idx[(domain_name, lbl)] for lbl in labels])

        if self.method == 'derpp' and self.model is not None:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()

        logits_for_memory = None
        if self.method == 'derpp' and self.old_model is not None:
            logits_for_memory = []
            temp_loader = DataLoader(SequenceDataset(features_list, new_global_labels),
                                     batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
            with torch.no_grad():
                for src, mask, _, _ in temp_loader:
                    out = self.old_model(src, src_key_padding_mask=~mask)
                    logits_for_memory.append(out.cpu())
            logits_for_memory = torch.cat(logits_for_memory, dim=0)
            self.memory_logits.extend([logits_for_memory[i].numpy() for i in range(len(logits_for_memory))])

        self._update_memory(features_list, new_global_labels, logits_for_memory)

        if self.method == 'joint':
            train_x = self.joint_data_x
            train_y = np.array(self.joint_data_y)
            is_replay = [False] * len(train_x)
        elif self.method == 'gdumb' and len(self.memory_y) > 0:
            train_x = self.memory_x
            train_y = np.array(self.memory_y)
            is_replay = [True] * len(train_x)
        else:
            if self.method in ['er', 'derpp']:
                train_x = features_list + self.memory_x
                train_y = np.concatenate([new_global_labels, self.memory_y])
                is_replay = [False] * len(features_list) + [True] * len(self.memory_x)
            else:
                train_x = features_list
                train_y = new_global_labels
                is_replay = [False] * len(features_list)

        dataset = SequenceDataset(train_x, train_y, is_replay)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        if self.method == 'ewc' and len(new_pairs) > 0:
            task_loader = DataLoader(SequenceDataset(features_list, new_global_labels),
                                     batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            self._compute_ewc_fisher(task_loader)

        if self.method == 'si':
            self.si_accumulated_grad = {name: torch.zeros_like(p) for name, p in self.model.named_parameters()}

        # 根据方法动态设定 epoch 数
        if self.method == 'fine_tune':
            epochs = 20
        elif self.method == 'joint':
            epochs = 100
        else:
            epochs = 15

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for src, mask, batch_y, batch_is_replay in dataloader:
                self.optimizer.zero_grad()
                logits = self.model(src, src_key_padding_mask=~mask)
                loss = self.criterion(logits, batch_y)

                if self.method == 'ewc' and self.ewc_fisher:
                    ewc_loss = 0
                    for name, p in self.model.named_parameters():
                        if name in self.ewc_fisher:
                            ewc_loss += (self.ewc_fisher[name] * (p - self.ewc_old_params[name]).pow(2)).sum()
                    loss += (EWC_LAMBDA / 2) * ewc_loss

                elif self.method == 'si' and self.si_omega:
                    si_loss = 0
                    for name, p in self.model.named_parameters():
                        if name in self.si_omega:
                            si_loss += (self.si_omega[name] * (p - self.si_old_params[name]).pow(2)).sum()
                    loss += SI_LAMBDA * si_loss

                elif self.method == 'derpp' and self.old_model is not None:
                    replay_mask = batch_is_replay
                    if replay_mask.any():
                        replay_indices = replay_mask.nonzero(as_tuple=True)[0]
                        if len(replay_indices) > 0:
                            with torch.no_grad():
                                src_replay = src[:, replay_indices]
                                mask_replay = mask[replay_indices]
                                old_logits = self.old_model(src_replay, src_key_padding_mask=~mask_replay)
                            logits_replay = logits[replay_indices]
                            distill_loss = F.kl_div(
                                F.log_softmax(logits_replay[:, :old_logits.size(1)], dim=1),
                                F.softmax(old_logits, dim=1),
                                reduction='batchmean'
                            )
                            loss += DERPP_ALPHA * distill_loss

                loss.backward()

                if self.method == 'si':
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            self.si_accumulated_grad[name] += p.grad.abs().detach()

                self.optimizer.step()
                total_loss += loss.item()

        if self.method == 'si':
            self._update_si_omega()
            self.si_accumulated_grad = None

    def predict(self, features_list, domain_name):
        self.model.eval()
        dataset = SequenceDataset(features_list, np.zeros(len(features_list)))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        valid_global_indices = []
        valid_original_labels = []
        for (dom, orig_lbl), g_idx in self.class_to_global_idx.items():
            if dom == domain_name:
                valid_global_indices.append(g_idx)
                valid_original_labels.append(orig_lbl)

        all_preds = []
        with torch.no_grad():
            for src, mask, _, _ in dataloader:
                logits = self.model(src, src_key_padding_mask=~mask)
                domain_logits = logits[:, valid_global_indices]
                pred_local_idx = torch.argmax(domain_logits, dim=1).cpu().numpy()
                all_preds.extend(pred_local_idx)

        pred_labels = np.array([valid_original_labels[idx] for idx in all_preds])
        return pred_labels

# ====================== 评估指标 ======================
def compute_accuracy(pred_labels, true_labels):
    return np.mean(pred_labels == true_labels)

def compute_acc_bwt(acc_matrix):
    """
    从下三角矩阵 acc_matrix (T x T) 计算 ACC 和 BWT。
    acc_matrix[i, j] 表示学完第 i+1 个域后在第 j+1 个域上的准确率。
    """
    T = acc_matrix.shape[0]
    # ACC = 最后一行（学完所有域）的平均准确率
    ACC = np.mean(acc_matrix[T-1, :])
    # BWT = 平均遗忘（最后一行与对角线之差，仅前 T-1 个域）
    bwt_sum = 0.0
    for j in range(T-1):
        bwt_sum += (acc_matrix[T-1, j] - acc_matrix[j, j])
    BWT = bwt_sum / (T-1) if T > 1 else 0.0
    return ACC, BWT

def parse_matrix_from_log(log_file):
    """从已有日志文件中提取准确率矩阵（用于离线解析）"""
    if not os.path.exists(log_file):
        return None
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_idx = None
    for i in range(len(lines)-1, -1, -1):
        if 'Accuracy Matrix:' in lines[i]:
            start_idx = i + 1
            break
    if start_idx is None:
        return None

    matrix = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line.startswith('After'):
            break
        parts = line.split(':')
        if len(parts) < 2:
            continue
        nums_str = parts[1].strip()
        nums = [float(x) for x in nums_str.split()]
        matrix.append(nums)
    return np.array(matrix) if len(matrix) == 3 else None  # 3个域

# ====================== 单个顺序 + 单个方法的评估 ======================
def evaluate_crossdomain(task_order, method, log_file):
    """对给定的任务顺序和方法进行完整评估，输出到指定的日志文件，返回 ACC, BWT, acc_matrix"""
    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee

    try:
        print(f"\n{'='*60}\nDomain-IL Evaluation (Transformer, pooling={POOLING_MODE}, method={method})\n{'='*60}")
        print(f"Task order: {' -> '.join(task_order)}")

        print("\nPreloading validation sets...")
        val_data_cache = {}
        for domain in task_order:
            val_feat, val_lab = load_domain_data(domain, phase='val')
            val_data_cache[domain] = (val_feat, val_lab)
            print(f"  {domain}: {len(val_feat)} videos")

        first_train_feat, _ = load_domain_data(task_order[0], phase='train')
        input_dim = first_train_feat[0].shape[-1]

        T = len(task_order)
        acc_matrix = np.zeros((T, T))
        classifier = TrainableTransformerClassifier(input_dim=input_dim, method=method, device=DEVICE)

        for i, domain in enumerate(task_order):
            print(f"\n--- Domain {i+1}: {domain} ---")
            train_feat, train_lab = load_domain_data(domain, phase='train')
            print(f"  Training videos: {len(train_feat)}, Classes: {np.unique(train_lab)}")

            classifier.update(train_feat, train_lab, domain_name=domain)

            for j in range(i + 1):
                eval_domain = task_order[j]
                val_feat, val_lab = val_data_cache[eval_domain]
                pred_labels = classifier.predict(val_feat, domain_name=eval_domain)
                acc = compute_accuracy(pred_labels, val_lab)
                acc_matrix[i, j] = acc
                print(f"  Accuracy on {eval_domain}: {acc:.4f}")

        ACC, BWT = compute_acc_bwt(acc_matrix)

        print("\n" + "="*60)
        print("Accuracy Matrix (rows: after domain i, cols: domain j):")
        print(acc_matrix)
        print("\nMetrics:")
        print(f"  ACC = {ACC:.4f}")
        print(f"  BWT = {BWT:.4f}")
        print("="*60)

        print("\n\n========== Final Results ==========")
        print(f"Method: {method}")
        print(f"ACC : {ACC:.4f}")
        print(f"BWT : {BWT:.4f}")
        print("\nAccuracy Matrix:")
        for i, row in enumerate(acc_matrix):
            print(f"  After {task_order[i]:12}: " + "  ".join([f"{v:.4f}" for v in row]))

        return ACC, BWT, acc_matrix
    except Exception as e:
        print(f"Error running {method} on order {task_order}: {e}")
        raise
    finally:
        sys.stdout = original_stdout
        tee.close()

# ====================== 运行单个顺序的所有方法 ======================
def run_order(task_order, methods=None, skip_existing=True):
    """对给定顺序运行所有方法，返回 {method: (ACC, BWT)}"""
    if methods is None:
        methods = ['fine_tune', 'joint', 'ewc', 'si', 'er', 'gdumb', 'derpp']

    order_str = '_'.join(task_order)
    results = {}

    for method in methods:
        log_filename = f"crossdomain_transformer_{method}_{order_str}_{TRAIN_EPOCHS}.txt"
        if skip_existing and os.path.exists(log_filename):
            # 直接从日志解析矩阵计算 ACC/BWT
            matrix = parse_matrix_from_log(log_filename)
            if matrix is not None:
                ACC, BWT = compute_acc_bwt(matrix)
                print(f"Log file exists: {log_filename}. Parsed ACC={ACC:.4f}, BWT={BWT:.4f}")
                results[method] = (ACC, BWT)
                continue
            else:
                print(f"Failed to parse matrix from {log_filename}, will retrain.")

        print(f"\n\n{'#'*60}")
        print(f"# Running {method} on order: {' -> '.join(task_order)}")
        print(f"# Log will be saved to {log_filename}")
        print(f"{'#'*60}\n")
        try:
            ACC, BWT, _ = evaluate_crossdomain(task_order, method, log_filename)
            results[method] = (ACC, BWT)
        except Exception as e:
            print(f"Failed {method} on {order_str}: {e}")
            results[method] = None
    return results

# ====================== 生成 CSV 汇总 ======================
def save_order_csv(task_order, results, methods_order):
    """保存单个顺序的 CSV 文件，如 activitynet_ucf101_fcvid_acc_bwt.csv"""
    order_str = '_'.join(task_order)
    csv_file = f"{order_str}_acc_bwt_{TRAIN_EPOCHS}.csv"
    with open(csv_file, 'w') as f:
        f.write("Method,ACC,BWT\n")
        for method in methods_order:
            if method in results and results[method] is not None:
                acc, bwt = results[method]
                f.write(f"{method},{acc:.4f},{bwt:.4f}\n")
            else:
                f.write(f"{method},NA,NA\n")
    print(f"Saved CSV: {csv_file}")

# ====================== 主函数：运行所有排列 ======================
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.file.write(msg)
        self.stdout.write(msg)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

def main():
    domains = ["activitynet", "ucf101", "fcvid"]
    all_orders = list(itertools.permutations(domains))
    methods_order = ['fine_tune', 'joint', 'ewc', 'si', 'er', 'gdumb', 'derpp']

    for order in all_orders:
        print(f"\n\n{'='*80}")
        print(f"Processing order: {' -> '.join(order)}")
        print(f"{'='*80}")
        results = run_order(order, methods=methods_order, skip_existing=True)
        save_order_csv(order, results, methods_order)

    print("\nAll orders completed.")

if __name__ == '__main__':
    main()