import os
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

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
    "activitynet": {"val": "val.txt"},
    "ucf101":      {"val": "val.txt"},
    "fcvid":       {"val": "fcv_val.txt"}
}

DOMAIN_DATASET_TYPES = {
    "activitynet": "actnet",
    "ucf101": "ucf",
    "fcvid": "fcvid"
}

NUM_SELECTED_CLASSES = 10
SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42
OUTPUT_PREFIX = "tsne"

UNIFIED_CLASS_NAMES = {
    0: "Makeup & Beauty Care",
    1: "Archery",
    2: "Cycling & Biking",
    3: "Dance",
    4: "Hair Care",
    5: "Oral & Facial Cleaning",
    6: "Outdoor Recreation & Entertainment",
    7: "Extreme Aerial Sports",
    8: "Animal Riding & Competition",
    9: "Paddle Water Sports",
    10: "Combat & Martial Arts",
    11: "Boxing",
    12: "Drill & Team Performance",
    13: "Strength Training & Fitness",
    14: "Cleaning & Housework",
    15: "Pet Care & Dog Walking",
    16: "Ice & Snow Sports",
    17: "Festivals & Celebrations",
    18: "Frisbee Sports",
    19: "Fencing",
    20: "Cooking & Food Preparation",
    21: "Beverage Making",
    22: "Table Tennis",
    23: "Diving",
    24: "String Instrument Performance",
    25: "Wind Instrument Performance",
    26: "Percussion Instrument Performance",
    27: "Keyboard Instrument Performance",
    28: "Football / Soccer",
    29: "Racket Sports",
    30: "Bowling",
    31: "Rock Climbing & Rope Climbing",
    32: "Rope Skipping",
    33: "Surfing & Water Board Sports",
    34: "Shaving & Body Care",
    35: "Swimming",
    36: "Tai Chi",
    37: "Gymnastics",
    38: "Knitting & Handicrafts",
    39: "Skateboarding & Skating",
    40: "Basketball",
    41: "Billiards & Board Games",
    42: "Painting & Writing"
}

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

def load_features_from_h5(h5_path, video_names, aggregate=True):
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
                vec = np.zeros(768, dtype=np.float32)
            else:
                vec = grp['vectors'][:]
            if aggregate and vec.ndim == 2:
                vec = vec.mean(axis=0)
            features.append(vec.astype(np.float32))
    return np.array(features)

def get_domain_all_labels(domain_name, phase='val'):
    split_dir = DOMAIN_SPLIT_DIRS[domain_name]
    file_name = DOMAIN_FILE_NAMES[domain_name][phase]
    txt_file = os.path.join(split_dir, file_name)
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Missing file: {txt_file}")
    dataset_type = DOMAIN_DATASET_TYPES[domain_name]
    video_names, all_labels = load_split_file(txt_file, dataset_type)
    return video_names, all_labels

def load_domain_data_for_classes(domain_name, selected_classes, phase='val', samples_per_class=50):
    split_dir = DOMAIN_SPLIT_DIRS[domain_name]
    file_name = DOMAIN_FILE_NAMES[domain_name][phase]
    txt_file = os.path.join(split_dir, file_name)
    dataset_type = DOMAIN_DATASET_TYPES[domain_name]
    video_names, all_labels = load_split_file(txt_file, dataset_type)

    sampled_indices = []
    sampled_class_labels = []
    np.random.seed(RANDOM_SEED)
    for cls in selected_classes:
        cls_indices = np.where(all_labels == cls)[0]
        if len(cls_indices) == 0:
            print(f"Warning: Class {cls} not found in {domain_name}, skipping.")
            continue
        if len(cls_indices) > samples_per_class:
            chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            chosen = cls_indices
        sampled_indices.extend(chosen)
        sampled_class_labels.extend([cls] * len(chosen))

    sampled_video_names = [video_names[i] for i in sampled_indices]
    h5_file = DOMAIN_H5_FILES[domain_name]
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Missing H5 file: {h5_file}")
    features = load_features_from_h5(h5_file, sampled_video_names, aggregate=True)
    return features, np.array(sampled_class_labels)

def load_mmd_values(file_path="cross_domain_class_mmd_cka.txt"):
    mmd_dict = {
        "act-ucf": {},
        "act-fcv": {},
        "ucf-fcv": {}
    }
    current_pair = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("处理域对: activitynet vs ucf101"):
                current_pair = "act-ucf"
            elif line.startswith("处理域对: activitynet vs fcvid"):
                current_pair = "act-fcv"
            elif line.startswith("处理域对: ucf101 vs fcvid"):
                current_pair = "ucf-fcv"
            elif line.startswith("Class") and current_pair:
                parts = line.split(":")
                class_idx = int(parts[0].split()[1])
                mmd_val = float(parts[1].strip())
                mmd_dict[current_pair][class_idx] = mmd_val
    return mmd_dict

def main():
    print("Step 1: Loading label sets from all domains to find common classes...")
    domains = ["activitynet", "ucf101", "fcvid"]
    domain_labels_sets = {}
    for domain in domains:
        _, labels = get_domain_all_labels(domain, phase='val')
        domain_labels_sets[domain] = set(np.unique(labels))
        print(f"  {domain}: {len(domain_labels_sets[domain])} unique classes")

    common_classes = set.intersection(*domain_labels_sets.values())
    common_classes = np.array(sorted(common_classes))
    print(f"Common classes across all domains: {len(common_classes)}")

    if len(common_classes) == 0:
        print("No common classes found! Exiting.")
        return

    desired_class_indices = [
        4,   # Hair Care
        5,   # Oral & Facial Cleaning
        9,   # Paddle Water Sports
        13,  # Strength Training & Fitness
        25,  # Wind Instrument Performance
        26,  # Percussion Instrument Performance
        35,  # Swimming
        37,  # Gymnastics
        38,  # Knitting & Handicrafts
        40   # Basketball
    ]
    selected_classes = np.array([cls for cls in desired_class_indices if cls in common_classes])
    if len(selected_classes) < len(desired_class_indices):
        missing = set(desired_class_indices) - set(selected_classes)
        print(f"Warning: These classes are not common across all domains: {missing}")
    print(f"Selected {len(selected_classes)} classes (manually chosen for optimized legend): {selected_classes.tolist()}")

    mmd_values = load_mmd_values("cross_domain_class_mmd_cka.txt")
    print("\nSelected classes and their MMD values between domain pairs:")
    print("Class Index | Class Name                             | Act-UCF MMD | Act-FCV MMD | UCF-FCV MMD")
    print("-" * 90)
    for cls in selected_classes:
        name = UNIFIED_CLASS_NAMES.get(cls, f"Class {cls}")
        a_u = mmd_values["act-ucf"].get(cls, 0.0)
        a_f = mmd_values["act-fcv"].get(cls, 0.0)
        u_f = mmd_values["ucf-fcv"].get(cls, 0.0)
        print(f"{cls:11d} | {name:40s} | {a_u:.4f}      | {a_f:.4f}      | {u_f:.4f}")

    for domain in domains:
        print(f"\n{'='*50}")
        print(f"Processing {domain}...")
        features, class_labels = load_domain_data_for_classes(
            domain, selected_classes, phase='val', samples_per_class=SAMPLES_PER_CLASS
        )
        if len(features) == 0:
            print(f"  No samples loaded for {domain}, skipping.")
            continue
        print(f"  Loaded {len(features)} samples from {len(np.unique(class_labels))} classes.")

        print(f"  Running t‑SNE for {domain}...")
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000, verbose=1)
        X_2d = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        unique_classes_in_plot = np.unique(class_labels)
        num_classes = len(unique_classes_in_plot)
        colors = cm.tab10(np.linspace(0, 1, num_classes))

        for i, cls in enumerate(unique_classes_in_plot):
            idx = np.where(class_labels == cls)[0]
            label_name = UNIFIED_CLASS_NAMES.get(cls, f"Class {cls}")
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                        c=[colors[i]], label=label_name,
                        alpha=0.8, s=50, edgecolors='none')

        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=False,
                   ncol=2 if num_classes > 8 else 1)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        output_file = f"{OUTPUT_PREFIX}_{domain}.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  Plot saved to {output_file}")
        plt.close()

    print("\nAll plots generated successfully.")

if __name__ == '__main__':
    main()