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

# ====================== 合成域配置 ======================
TRAIN_LIST_FILE = 'train.txt'
VAL_LIST_FILE   = 'val.txt'

SYNTHETIC_H5_PATHS = [
    'act_vit.h5',
    'act_synthetic_1.h5',
    'act_synthetic_2.h5',
    'act_synthetic_3.h5'
]

DOMAIN_NAMES = ["act_vit", "syn1", "syn2", "syn3"]
NUM_DOMAINS = len(SYNTHETIC_H5_PATHS)


# ========== Transformer 参数 ==========
D_MODEL = 768
NHEAD = 8
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
POOLING_MODE = 'cls'

# ========== 训练参数 ==========
TRAIN_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_FILE = f"cross_domain_syn_44matrix{TRAIN_EPOCHS}_200_class.txt"

# ====================== 数据加载函数 ======================
def load_video_list(txt_path, dataset_type='actnet'):
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
        for name in tqdm(video_names, desc=f"Loading {os.path.basename(h5_path)}", leave=False):
            grp = f.get(name)
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

def load_domain_data(domain_idx, phase='train'):
    list_file = TRAIN_LIST_FILE if phase == 'train' else VAL_LIST_FILE
    h5_path = SYNTHETIC_H5_PATHS[domain_idx]
    video_names, labels = load_video_list(list_file, 'actnet')
    features = load_features_from_h5(h5_path, video_names, aggregate=False)
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

# ====================== Transformer 模型 ======================
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
def train_on_domain(domain_idx):
    h5_path = SYNTHETIC_H5_PATHS[domain_idx]
    print(f"\n{'='*60}\nTraining on Domain {DOMAIN_NAMES[domain_idx]}\n{'='*60}")

    train_feat, train_lab = load_domain_data(domain_idx, phase='train')
    unique_labels = np.unique(train_lab)
    num_classes = len(unique_labels)
    class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_class = {i: lbl for i, lbl in enumerate(unique_labels)}

    mapped_labels = np.array([class_to_idx[l] for l in train_lab])

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

def evaluate_on_domain(model, target_domain_idx, class_to_idx_source):
    val_feat, val_lab = load_domain_data(target_domain_idx, phase='val')
    # 过滤出源域训练过的类别（因为类别空间一致，全部保留）
    valid_mask = np.isin(val_lab, list(class_to_idx_source.keys()))
    if not np.any(valid_mask):
        print(f"  No overlapping classes with source, accuracy = 0.0")
        return 0.0

    val_feat_filtered = [val_feat[i] for i in range(len(val_feat)) if valid_mask[i]]
    val_lab_filtered = val_lab[valid_mask]
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
    print(f"\n{'='*60}\nCross-Domain Generalization Test on Synthetic Domains\n{'='*60}")

    acc_matrix = np.zeros((NUM_DOMAINS, NUM_DOMAINS))

    for src_idx in range(NUM_DOMAINS):
        model, class_to_idx, _ = train_on_domain(src_idx)

        print(f"\nEvaluating model trained on {DOMAIN_NAMES[src_idx]}:")
        for tgt_idx in range(NUM_DOMAINS):
            acc = evaluate_on_domain(model, tgt_idx, class_to_idx)
            acc_matrix[src_idx, tgt_idx] = acc
            print(f"  Accuracy on {DOMAIN_NAMES[tgt_idx]}: {acc:.4f}")

    # 计算平均跨域下降
    print("\n" + "="*60)
    print("Accuracy Matrix (rows: train domain, cols: test domain):")
    print(acc_matrix)
    print("\nCross-Domain Drop Analysis:")
    drops = []
    for i, src_name in enumerate(DOMAIN_NAMES):
        src_acc = acc_matrix[i, i]
        for j, tgt_name in enumerate(DOMAIN_NAMES):
            if i != j:
                drop = src_acc - acc_matrix[i, j]
                drops.append(drop)
                print(f"  {src_name} -> {tgt_name}: {drop:.4f} drop")
    avg_drop = np.mean(drops) if drops else 0.0
    print(f"\nAverage Cross-Domain Drop: {avg_drop:.4f}")

    return acc_matrix, avg_drop

# ====================== 日志与主函数 ======================
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
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    try:
        acc_matrix, avg_drop = run_cross_domain_evaluation()

        print("\n\n========== Final Cross-Domain Generalization Summary ==========")
        header = "Train\\Test     | " + " | ".join([f"{name:<8}" for name in DOMAIN_NAMES])
        print(header)
        print("-" * (15 + 11 * NUM_DOMAINS))
        for i, src_name in enumerate(DOMAIN_NAMES):
            row_str = f"{src_name:<15} | " + " | ".join([f"{acc_matrix[i,j]:.4f}   " for j in range(NUM_DOMAINS)])
            print(row_str)
        print(f"\nAverage Cross-Domain Drop: {avg_drop:.4f}")
    finally:
        sys.stdout = tee.stdout
        tee.close()

if __name__ == '__main__':
    main()