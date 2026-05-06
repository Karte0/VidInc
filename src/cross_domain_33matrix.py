import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
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

SPLIT_BASE = ""

DOMAINS = ["activitynet", "ucf101", "fcvid"]

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
POOLING_MODE = 'cls'         # 'cls' 或 'mean'

# ========== 训练参数 ==========
TRAIN_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_FILE = f"cross_domain_33matrix{TRAIN_EPOCHS}.txt"
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
    def __init__(self, features_list, labels):
        self.features = features_list
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label

def collate_fn(batch):
    features, labels = zip(*batch)
    padded_feats = pad_sequence(features, batch_first=False)
    lengths = torch.tensor([f.size(0) for f in features])
    mask = torch.zeros(padded_feats.size(1), padded_feats.size(0), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    mask = mask.to(DEVICE)
    labels = torch.stack(labels).to(DEVICE)
    return padded_feats.to(DEVICE), mask, labels

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

# ====================== 单域训练与评估 ======================
def train_on_domain(source_domain):
    """在指定源域上训练模型，返回训练好的模型和源域的类别映射"""
    print(f"\n{'='*60}\nTraining on {source_domain}\n{'='*60}")

    # 加载训练数据
    train_feat, train_lab = load_domain_data(source_domain, phase='train')
    unique_labels = np.unique(train_lab)
    num_classes = len(unique_labels)
    class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_class = {i: lbl for i, lbl in enumerate(unique_labels)}

    # 映射标签为连续索引
    mapped_labels = np.array([class_to_idx[l] for l in train_lab])

    # 创建模型
    input_dim = train_feat[0].shape[-1]
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pooling=POOLING_MODE
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    dataset = SequenceDataset(train_feat, mapped_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(TRAIN_EPOCHS):
        total_loss = 0.0
        for src, mask, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(src, src_key_padding_mask=~mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{TRAIN_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    return model, class_to_idx, idx_to_class

def evaluate_on_domain(model, target_domain, class_to_idx_source, idx_to_class_source):
    """在目标域上评估模型，由于标签空间不同，只计算模型能预测的那些类别（若源域和目标域标签有交集）"""
    val_feat, val_lab = load_domain_data(target_domain, phase='val')
    # 找出目标域中存在于源域训练类别中的样本
    valid_mask = np.isin(val_lab, list(class_to_idx_source.keys()))
    if not np.any(valid_mask):
        print(f"  No overlapping classes between source and {target_domain}, accuracy set to 0.0")
        return 0.0

    val_feat_filtered = [val_feat[i] for i in range(len(val_feat)) if valid_mask[i]]
    val_lab_filtered = val_lab[valid_mask]

    # 映射目标标签到源域索引
    mapped_labels = np.array([class_to_idx_source[l] for l in val_lab_filtered])

    dataset = SequenceDataset(val_feat_filtered, mapped_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, mask, batch_y in dataloader:
            logits = model(src, src_key_padding_mask=~mask)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc

def run_cross_domain_evaluation():
    print(f"\n{'='*60}\nCross-Domain Generalization Test (Transformer, pooling={POOLING_MODE})\n{'='*60}")

    # 预加载所有域的验证集数据（仅用于统计）
    val_info = {}
    for domain in DOMAINS:
        _, val_lab = load_domain_data(domain, phase='val')
        val_info[domain] = val_lab

    # 结果矩阵：行=源域，列=目标域
    acc_matrix = np.zeros((len(DOMAINS), len(DOMAINS)))

    for i, source_domain in enumerate(DOMAINS):
        # 训练源域模型
        model, class_to_idx, idx_to_class = train_on_domain(source_domain)

        # 在所有目标域上评估
        print(f"\nEvaluating model trained on {source_domain}:")
        for j, target_domain in enumerate(DOMAINS):
            acc = evaluate_on_domain(model, target_domain, class_to_idx, idx_to_class)
            acc_matrix[i, j] = acc
            print(f"  Accuracy on {target_domain}: {acc:.4f}")

    # 计算平均跨域下降
    print("\n" + "="*60)
    print("Accuracy Matrix (rows: train domain, cols: test domain):")
    print(acc_matrix)
    print("\nCross-Domain Drop Analysis:")
    drops = []
    for i, src in enumerate(DOMAINS):
        src_acc = acc_matrix[i, i]
        for j, tgt in enumerate(DOMAINS):
            if i != j:
                drop = src_acc - acc_matrix[i, j]
                drops.append(drop)
                print(f"  {src} -> {tgt}: {drop:.4f} drop")
    avg_drop = np.mean(drops) if drops else 0.0
    print(f"\nAverage Cross-Domain Drop: {avg_drop:.4f}")

    return acc_matrix, avg_drop

def main():
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

    tee = Tee(LOG_FILE)
    sys.stdout = tee

    try:
        acc_matrix, avg_drop = run_cross_domain_evaluation()

        print("\n\n========== Final Cross-Domain Generalization Summary ==========")
        print("Train\\Test     | " + " | ".join([f"{d[:8]:<8}" for d in DOMAINS]))
        print("-" * (15 + 11 * len(DOMAINS)))
        for i, src in enumerate(DOMAINS):
            row_str = f"{src:<15} | " + " | ".join([f"{acc_matrix[i,j]:.4f}   " for j in range(len(DOMAINS))])
            print(row_str)
        print(f"\nAverage Cross-Domain Drop: {avg_drop:.4f}")
    finally:
        sys.stdout = tee.stdout
        tee.close()

if __name__ == '__main__':
    main()