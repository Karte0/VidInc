import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.spatial.distance import pdist
from itertools import combinations

# ====================== 配置 ======================
H5_BASE_DIR = ""

DOMAIN_H5_FILES = {
    "activitynet": os.path.join(H5_BASE_DIR, "act_vit.h5"),
    "ucf101":      os.path.join(H5_BASE_DIR, "ucf_vit_25.h5"),
    "fcvid":       os.path.join(H5_BASE_DIR, "fcvid_vit_train.h5")
}

SPLIT_BASE = ""

DOMAIN_SPLIT_DIRS = {
    "activitynet": os.path.join(SPLIT_BASE, "Activitynet"),
    "ucf101":      os.path.join(SPLIT_BASE, "ucf101"),
    "fcvid":       os.path.join(SPLIT_BASE, "fcvid")
}

DOMAIN_FILE_NAMES = {
    "activitynet": {"train": "train.txt", "val": "val.txt", "test": "test.txt"},
    "ucf101":      {"train": "train.txt", "val": "val.txt", "test": "test.txt"},
    "fcvid":       {"train": "fcv_train.txt", "val": "fcv_val.txt", "test": "fcv_test.txt"}
}

DOMAIN_DATASET_TYPES = {
    "activitynet": "actnet",
    "ucf101":      "ucf",
    "fcvid":       "fcvid"
}

PHASES = ('train', 'val', 'test')          # 整合所有可用阶段

# ========== 特征提取配置 ==========
AGGREGATE_VIDEO = False                    # False: 帧级特征，每帧一个样本
FEATURE_DIM = 768

# ========== MMD 参数 ==========
MMD_M = 10000

# ========== CKA 参数 ==========
# 若为整数，则从每个域随机采样指定数量的帧；若为 None，则使用全部帧（需注意内存！）
CKA_N_SAMPLES = 1000   # 设为 None 则全量计算 CKA

RANDOM_SEED = 42
OUTPUT_FILE = "cross_domain_class_mmd_cka_frame_level.txt"


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
            else:   # ucf / fcvid
                if len(parts) >= 2:
                    video_names.append(parts[0])
                    labels.append(int(parts[1]))
    return video_names, np.array(labels)


def load_domain_data(domain_name, phases=PHASES):
    """
    加载域的全部数据（按指定阶段），返回：
        - features: (N_frames, D) 的帧级特征数组
        - labels:   (N_frames,) 的帧级标签数组（每个视频的标签复制到其所有帧）
    """
    split_dir = DOMAIN_SPLIT_DIRS[domain_name]
    h5_path = DOMAIN_H5_FILES[domain_name]
    dataset_type = DOMAIN_DATASET_TYPES[domain_name]

    all_video_names = []
    all_labels_per_video = []

    # 1. 收集所有视频名及其标签
    for phase in phases:
        file_name = DOMAIN_FILE_NAMES[domain_name].get(phase)
        if file_name is None:
            continue
        txt_file = os.path.join(split_dir, file_name)
        if not os.path.exists(txt_file):
            print(f"  注意：{txt_file} 不存在，跳过阶段 {phase}")
            continue
        names, labs = load_split_file(txt_file, dataset_type)
        all_video_names.extend(names)
        all_labels_per_video.extend(labs)

    if len(all_video_names) == 0:
        raise ValueError(f"{domain_name} 未收集到任何视频。")

    print(f"  {domain_name}: {len(all_video_names)} 个视频")

    # 2. 加载 H5 特征，并为每一帧赋予对应视频的标签
    frame_features = []
    frame_labels = []
    with h5py.File(h5_path, 'r') as f:
        for name, label in tqdm(zip(all_video_names, all_labels_per_video),
                                total=len(all_video_names),
                                desc=f"Loading {os.path.basename(h5_path)}", leave=False):
            # 尝试多种可能的键名
            grp = f.get(name)
            if grp is None:
                base = os.path.basename(name)
                base_no_ext = os.path.splitext(base)[0]
                grp = f.get(base_no_ext) or f.get(base)
            if grp is None:
                continue   # 静默跳过缺失视频
            vectors = grp['vectors'][()]          # (T, 768)
            T = vectors.shape[0]
            frame_features.append(vectors.astype(np.float32))
            frame_labels.extend([label] * T)

    if not frame_features:
        return np.empty((0, FEATURE_DIM)), np.array([])
    return np.vstack(frame_features), np.array(frame_labels)


# ====================== 距离计算函数 ======================
def linear_time_mmd(X, Y, m=MMD_M, sigma=None):
    """线性时间 MMD 估计（无偏），使用 RBF 核"""
    n_x, n_y = len(X), len(Y)
    m = min(m, n_x // 2, n_y // 2)
    if m == 0:
        return 0.0

    idx_x = np.random.choice(n_x, size=2 * m, replace=True)
    idx_y = np.random.choice(n_y, size=2 * m, replace=True)
    x_pair1 = X[idx_x[:m]]
    x_pair2 = X[idx_x[m:2 * m]]
    y_pair1 = Y[idx_y[:m]]
    y_pair2 = Y[idx_y[m:2 * m]]

    if sigma is None:
        # 动态估计 RBF 核带宽（基于子样本中位数）
        max_samples = 2000
        if n_x > max_samples and n_y > max_samples:
            sample = np.vstack([X[:max_samples], Y[:max_samples]])
        else:
            sample = np.vstack([X, Y])
        distances = pdist(sample)
        sigma = np.median(distances)
        if sigma == 0:
            sigma = 1.0

    def rbf_kernel(a, b, sigma):
        diff = a - b
        return np.exp(-np.sum(diff ** 2, axis=1) / (2 * sigma ** 2))

    term1 = rbf_kernel(x_pair1, x_pair2, sigma).mean()
    term2 = rbf_kernel(y_pair1, y_pair2, sigma).mean()
    term3 = rbf_kernel(x_pair1, y_pair1, sigma).mean()
    mmd2 = term1 + term2 - 2 * term3
    return np.sqrt(max(mmd2, 0))


def centered_kernel_alignment(X, Y):
    """
    计算 Centered Kernel Alignment (线性核)，衡量两个特征表示空间的相似性。
    详细原理见代码后注释。
    """
    # 中心化特征
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # 线性核 Gram 矩阵
    K = X @ X.T
    L = Y @ Y.T

    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n   # 中心化矩阵
    Kc = H @ K @ H
    Lc = H @ L @ H

    # HSIC (Hilbert-Schmidt Independence Criterion) 的无偏估计
    hsic = np.trace(Kc @ Lc) / ((n - 1) ** 2)
    var1 = np.trace(Kc @ Kc) / ((n - 1) ** 2)
    var2 = np.trace(Lc @ Lc) / ((n - 1) ** 2)

    cka = hsic / (np.sqrt(var1 * var2) + 1e-10)
    return cka


# ====================== 主流程 ======================
def main():
    np.random.seed(RANDOM_SEED)

    print("加载所有域的视频特征（帧级）及类别标签...")
    print(f"特征模式: {'帧级' if not AGGREGATE_VIDEO else '视频级平均'}")
    print(f"加载阶段: {PHASES}")
    if CKA_N_SAMPLES is None:
        print("CKA 计算模式: 全量（使用所有帧，请注意内存消耗）")
    else:
        print(f"CKA 采样数: {CKA_N_SAMPLES} 帧/域")
    print()

    domains = ["activitynet", "ucf101", "fcvid"]
    raw_data = {}
    raw_labels = {}

    # 第一步：加载原始数据
    for domain_name in domains:
        try:
            print(f"处理域: {domain_name}")
            data, labels = load_domain_data(domain_name)
            raw_data[domain_name] = data
            raw_labels[domain_name] = labels
            print(f"  -> 总帧数: {len(data)}，原始标签范围: {labels.min()}~{labels.max()}\n")
        except Exception as e:
            print(f"加载 {domain_name} 失败: {e}，跳过该域。")
            continue

    if len(raw_data) < 2:
        print("至少需要两个域才能进行比较，退出。")
        return

    # 第二步：构建统一的标签映射（0 ~ N-1）
    all_original_labels = set()
    for labels in raw_labels.values():
        all_original_labels.update(np.unique(labels))
    sorted_labels = sorted(list(all_original_labels))
    num_classes = len(sorted_labels)
    print(f"实际类别数: {num_classes} (原始标签 {sorted_labels[:5]}...)")
    label_map = {orig: i for i, orig in enumerate(sorted_labels)}

    # 应用映射
    domain_data = {}
    domain_labels = {}
    for domain in domains:
        if domain not in raw_data:
            continue
        data = raw_data[domain]
        labels = np.array([label_map[l] for l in raw_labels[domain]])
        domain_data[domain] = data
        domain_labels[domain] = labels
        print(f"{domain} 映射后标签范围: {labels.min()}~{labels.max()}")

    domain_pairs = list(combinations(domain_data.keys(), 2))

    with open(OUTPUT_FILE, 'w') as f_out:
        f_out.write("Cross-domain Class-wise MMD and Overall CKA (Frame-level)\n")
        f_out.write(f"Loaded phases: {PHASES}\n")
        f_out.write(f"Feature mode: frame-level (each frame is a sample)\n")
        f_out.write(f"MMD parameter m = {MMD_M}\n")
        f_out.write(f"CKA sample setting: {'full data' if CKA_N_SAMPLES is None else f'{CKA_N_SAMPLES} frames per domain'}\n")
        f_out.write(f"Number of classes: {num_classes}\n")
        f_out.write("=" * 70 + "\n\n")

        for domain_a, domain_b in domain_pairs:
            print(f"\n{'='*50}")
            print(f"处理域对: {domain_a} vs {domain_b}")
            f_out.write(f"### {domain_a} vs {domain_b} ###\n")

            data_a = domain_data[domain_a]
            data_b = domain_data[domain_b]
            labels_a = domain_labels[domain_a]
            labels_b = domain_labels[domain_b]

            # ---------- 1. 类别 MMD ----------
            print("计算类别 MMD...")
            f_out.write(f"Class-wise MMD (mapped indices 0~{num_classes-1}):\n")
            for cls in range(num_classes):
                X_cls = data_a[labels_a == cls]
                Y_cls = data_b[labels_b == cls]
                if len(X_cls) == 0 or len(Y_cls) == 0:
                    mmd_val = np.nan
                else:
                    mmd_val = linear_time_mmd(X_cls, Y_cls, m=MMD_M)
                f_out.write(f"  Class {cls}: {mmd_val:.6f}\n")
                print(f"  Class {cls}: {mmd_val:.6f}")


    print(f"\n所有结果已保存至 {OUTPUT_FILE}")


if __name__ == '__main__':
    main()