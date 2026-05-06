import os
import sys
import json
import copy
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ====================== 配置 ======================
DATASET_CONFIGS = {
    'actnet': 'class_incremental/activityNet/Anet.json',
    'fcvid':  'class_incremental/fcvid/fcvid.json',
    'ucf':    'class_incremental/ucf101/ucf.json'
}
NUM_TASKS = 10
seed = 42
# 方法选择（将在主循环中被覆盖）
# METHOD = 'fine_tune'
# LOG_FILE = f'log_transformer_acc_bwt_{METHOD}.txt'

# ========== Transformer 参数 ==========
D_MODEL = 768
NHEAD = 8
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
POOLING_MODE = 'cls'

# ========== 训练参数 ==========
TRAIN_EPOCHS = 15_20_100  # 此常量仅用于日志文件名，实际训练 epoch 在 update 中动态确定
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 正则化 / 回放超参数 ==========
EWC_LAMBDA = 1000.0
SI_LAMBDA = 1.0
DERPP_ALPHA = 0.5
MEMORY_SIZE = 2000

# ====================== 数据加载函数 ======================
def load_video_list(txt_path, dataset_type):
    video_names = []
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if dataset_type == 'actnet' and len(parts) >= 3:
                video_names.append(parts[0])
                labels.append(int(parts[2]))
            elif dataset_type in ['fcvid', 'ucf'] and len(parts) >= 2:
                video_names.append(parts[0])
                labels.append(int(parts[1]))
    return video_names, np.array(labels)

def load_features_from_h5(h5_path, video_names, aggregate=False):
    features = []
    with h5py.File(h5_path, 'r') as f:
        for name in tqdm(video_names, desc="Loading features", leave=False):
            grp = f.get(name)
            if grp is None:
                print(f"Video {name} not found in H5 file")
                vec = np.zeros((1, D_MODEL), dtype=np.float32)
            else:
                vec = grp['vectors'][:]
            if aggregate:
                if vec.ndim == 2:
                    vec = vec.mean(axis=0)
            features.append(vec.astype(np.float32))
    return features

def load_task_data(config, task_id, phase='train'):
    dataset_type = config['dataset']
    if phase == 'train':
        list_dir = config['train_list']
    elif phase == 'val':
        list_dir = config['val_list']
    else:
        raise ValueError("phase must be 'train' or 'val'")

    txt_file = os.path.join(list_dir, f"{task_id}.txt")
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Missing file: {txt_file}")

    video_names, labels = load_video_list(txt_file, dataset_type)
    features = load_features_from_h5(config['Image'], video_names, aggregate=False)
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
    def __init__(self, input_dim, method=METHOD, device=DEVICE):
        self.input_dim = input_dim
        self.method = method
        self.device = device
        # 使用 (task_id, original_label) 作为唯一类别标识，避免标签冲突
        self.classes = []                 # (task_id, original_label)
        self.class_to_global_idx = {}     # (task_id, original_label) -> global_index
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.memory_x = []
        self.memory_y = []
        self.memory_logits = []

        self.joint_data_x = []
        self.joint_data_y = []

        self.ewc_fisher = {}
        self.ewc_old_params = {}

        self.si_omega = {}
        self.si_old_params = {}
        self.si_prev_params = None
        self.si_accumulated_grad = None

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
                new_model.classifier.weight[:old_num_classes] = self.model.classifier.weight
                new_model.classifier.bias[:old_num_classes] = self.model.classifier.bias
                new_model.input_proj.load_state_dict(self.model.input_proj.state_dict())
                new_model.pos_encoder.data.copy_(self.model.pos_encoder.data)
                if self.model.cls_token is not None:
                    new_model.cls_token.data.copy_(self.model.cls_token.data)
                new_model.transformer.load_state_dict(self.model.transformer.state_dict())

        self.model = new_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # 扩展 EWC/SI 状态字典的分类器部分
        def _extend_classifier_state(state_dict, old_dim, new_dim):
            if state_dict is None:
                return
            for name in list(state_dict.keys()):
                if 'classifier' in name:
                    tensor = state_dict[name]
                    if tensor.dim() == 2:
                        new_tensor = torch.zeros(new_dim, tensor.size(1), device=tensor.device, dtype=tensor.dtype)
                        new_tensor[:old_dim] = tensor
                        state_dict[name] = new_tensor
                    elif tensor.dim() == 1:
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

    def update(self, features_list, labels, task_id):
        """更新模型，task_id 用于构造唯一类别标识"""
        unique_labels = np.unique(labels)
        new_pairs = [(task_id, lbl) for lbl in unique_labels
                     if (task_id, lbl) not in self.class_to_global_idx]

        if len(new_pairs) == 0:
            print("  No new classes to add, skipping training.")
            return

        for pair in new_pairs:
            global_idx = len(self.classes)
            self.class_to_global_idx[pair] = global_idx
            self.classes.append(pair)

        self._expand_model([self.class_to_global_idx[p] for p in new_pairs])

        new_global_labels = np.array([self.class_to_global_idx[(task_id, lbl)] for lbl in labels])

        # DER++ 旧模型
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

    def predict(self, features_list, task_id):
        self.model.eval()
        dataset = SequenceDataset(features_list, np.zeros(len(features_list)))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # 找出属于当前 task_id 的全局索引
        valid_global_indices = []
        valid_original_labels = []
        for (tid, orig_lbl), g_idx in self.class_to_global_idx.items():
            if tid == task_id:
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

# ====================== 评估指标与主流程 ======================
def compute_accuracy(pred_labels, true_labels):
    return np.mean(pred_labels == true_labels)

def compute_acc_and_bwt(results_per_task):
    T = len(results_per_task)
    avg_acc_per_task = []
    for t in range(1, T + 1):
        seen_tasks = list(range(1, t + 1))
        acc_sum = sum(results_per_task[t][i] for i in seen_tasks)
        avg_acc = acc_sum / len(seen_tasks)
        avg_acc_per_task.append(avg_acc)
    ACC = np.mean(avg_acc_per_task)

    bwt_sum = 0.0
    for i in range(1, T):
        acc_initial = results_per_task[i][i]
        acc_final = results_per_task[T][i]
        bwt_sum += (acc_final - acc_initial)
    BWT = bwt_sum / (T - 1)
    return ACC, BWT

def evaluate_dataset(config_path, dataset_name):
    print(f"\n{'=' * 50}\nEvaluating {dataset_name} (method={METHOD})\n{'=' * 50}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if not config.get('Image'):
        raise ValueError(f"{dataset_name}: 'Image' path is empty.")
    if not config.get('train_list') or not config.get('val_list'):
        raise ValueError(f"{dataset_name}: 'train_list' or 'val_list' is empty.")

    print("Preloading validation sets...")
    val_data_cache = {}
    for task_id in range(1, NUM_TASKS + 1):
        val_feat, val_lab = load_task_data(config, task_id, phase='val')
        val_data_cache[task_id] = (val_feat, val_lab)

    first_train_feat, _ = load_task_data(config, 1, phase='train')
    input_dim = first_train_feat[0].shape[-1]

    results = {}
    classifier = TrainableTransformerClassifier(input_dim=input_dim, method=METHOD, device=DEVICE)

    for task_id in range(1, NUM_TASKS + 1):
        print(f"\n--- Task {task_id} ---")
        train_features, train_labels = load_task_data(config, task_id, phase='train')
        print(f"  Training videos: {len(train_features)}, New classes: {np.unique(train_labels)}")

        classifier.update(train_features, train_labels, task_id=task_id)

        acc_dict = {}
        for seen_id in range(1, task_id + 1):
            val_feat, val_lab = val_data_cache[seen_id]
            pred_labels = classifier.predict(val_feat, task_id=seen_id)
            acc = compute_accuracy(pred_labels, val_lab)
            acc_dict[seen_id] = acc
            print(f"  Accuracy on Task {seen_id} val set: {acc:.4f}")

        results[task_id] = acc_dict

    ACC, BWT = compute_acc_and_bwt(results)
    print(f"\n{dataset_name} Results:")
    print(f"  ACC = {ACC:.4f}")
    print(f"  BWT = {BWT:.4f}")
    return ACC, BWT

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    def write(self, msg):
        self.file.write(msg)
        self.stdout.write(msg)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

def run_all_methods():
    methods = ['fine_tune', 'joint', 'ewc', 'si', 'er', 'gdumb', 'derpp']
    #methods = ['joint']
    #methods = ['fine_tune', 'ewc', 'si', 'er', 'gdumb', 'derpp']
    final_summary = {}

    for method in methods:
        global METHOD, LOG_FILE
        METHOD = method
        LOG_FILE = f'log_transformer_acc_bwt_{METHOD}_{TRAIN_EPOCHS}_{seed}.txt'

        print(f"\n\n{'#'*60}")
        print(f"#  Running method: {METHOD}")
        print(f"{'#'*60}\n")

        tee = Tee(LOG_FILE)
        sys.stdout = tee

        try:
            all_results = {}
            for name, cfg_path in DATASET_CONFIGS.items():
                if not os.path.exists(cfg_path):
                    print(f"Warning: Config file {cfg_path} not found. Skipping {name}.")
                    continue
                try:
                    acc, bwt = evaluate_dataset(cfg_path, name)
                    all_results[name] = {'ACC': acc, 'BWT': bwt}
                except Exception as e:
                    import traceback
                    print(f"Error processing {name}: {e}")
                    traceback.print_exc()

            final_summary[method] = all_results

            print("\n\n========== Final Results for method: {} ==========".format(method))
            print("Dataset     |   ACC   |   BWT   ")
            print("------------|---------|---------")
            for name, res in all_results.items():
                print(f"{name:<11} | {res['ACC']:.4f} | {res['BWT']:.4f}")

        finally:
            sys.stdout = tee.stdout
            tee.close()

    # 打印跨方法总结表格
    print("\n\n" + "="*80)
    print("OVERALL SUMMARY: Class-Incremental Results Across All Methods")
    print("="*80)
    datasets = ['actnet', 'fcvid', 'ucf']
    print(f"{'Method':<12} | {'Dataset':<8} | {'ACC':>8} | {'BWT':>8}")
    print("-"*50)
    for method in methods:
        if method in final_summary:
            for ds in datasets:
                if ds in final_summary[method]:
                    acc = final_summary[method][ds]['ACC']
                    bwt = final_summary[method][ds]['BWT']
                    print(f"{method:<12} | {ds:<8} | {acc:8.4f} | {bwt:8.4f}")
                else:
                    print(f"{method:<12} | {ds:<8} | {'N/A':>8} | {'N/A':>8}")
        else:
            print(f"{method:<12} | {'ERROR':<8} | {'ERROR':>8} | {'ERROR':>8}")
    print("="*80)

def main():
    run_all_methods()

if __name__ == '__main__':
    main()