import os
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

H5_BASE_DIR = ''
DOMAIN_H5 = {
    'act_vit': os.path.join(H5_BASE_DIR, 'act_vit.h5'),
    'syn1':    os.path.join(H5_BASE_DIR, 'act_synthetic_1.h5'),
    'syn2':    os.path.join(H5_BASE_DIR, 'act_synthetic_2.h5'),
    'syn3':    os.path.join(H5_BASE_DIR, 'act_synthetic_3.h5'),
}

VAL_TXT = ''

NUM_RANDOM_CLASSES = 10
SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42
OUTPUT_PREFIX = 'tsne'

ACTIVITYNET_CLASS_NAMES = {
    0: "Applying sunscreen",
    1: "Archery",
    2: "Arm wrestling",
    3: "Assembling bicycle",
    4: "BMX",
    5: "Baking cookies",
    6: "Ballet",
    7: "Bathing dog",
    8: "Baton twirling",
    9: "Beach soccer",
    10: "Beer pong",
    11: "Belly dance",
    12: "Blow-drying hair",
    13: "Blowing leaves",
    14: "Braiding hair",
    15: "Breakdancing",
    16: "Brushing hair",
    17: "Brushing teeth",
    18: "Building sandcastles",
    19: "Bullfighting",
    20: "Bungee jumping",
    21: "Calf roping",
    22: "Camel ride",
    23: "Canoeing",
    24: "Capoeira",
    25: "Carving jack-o-lanterns",
    26: "Changing car wheel",
    27: "Cheerleading",
    28: "Chopping wood",
    29: "Clean and jerk",
    30: "Cleaning shoes",
    31: "Cleaning sink",
    32: "Cleaning windows",
    33: "Clipping cat claws",
    34: "Cricket",
    35: "Croquet",
    36: "Cumbia",
    37: "Curling",
    38: "Cutting the grass",
    39: "Decorating the Christmas tree",
    40: "Disc dog",
    41: "Discus throw",
    42: "Dodgeball",
    43: "Doing a powerbomb",
    44: "Doing crunches",
    45: "Doing fencing",
    46: "Doing karate",
    47: "Doing kickboxing",
    48: "Doing motocross",
    49: "Doing nails",
    50: "Doing step aerobics",
    51: "Drinking beer",
    52: "Drinking coffee",
    53: "Drum corps",
    54: "Elliptical trainer",
    55: "Fixing bicycle",
    56: "Fixing the roof",
    57: "Fun sliding down",
    58: "Futsal",
    59: "Gargling mouthwash",
    60: "Getting a haircut",
    61: "Getting a piercing",
    62: "Getting a tattoo",
    63: "Grooming dog",
    64: "Grooming horse",
    65: "Hammer throw",
    66: "Hand car wash",
    67: "Hand washing clothes",
    68: "Hanging wallpaper",
    69: "Having an ice cream",
    70: "High jump",
    71: "Hitting a pinata",
    72: "Hopscotch",
    73: "Horseback riding",
    74: "Hula hoop",
    75: "Hurling",
    76: "Ice fishing",
    77: "Installing carpet",
    78: "Ironing clothes",
    79: "Javelin throw",
    80: "Kayaking",
    81: "Kite flying",
    82: "Kneeling",
    83: "Knitting",
    84: "Laying tile",
    85: "Layup drill in basketball",
    86: "Long jump",
    87: "Longboarding",
    88: "Making a cake",
    89: "Making a lemonade",
    90: "Making a sandwich",
    91: "Making an omelette",
    92: "Mixing drinks",
    93: "Mooping floor",
    94: "Mowing the lawn",
    95: "Paintball",
    96: "Painting",
    97: "Painting fence",
    98: "Painting furniture",
    99: "Peeling potatoes",
    100: "Ping-pong",
    101: "Plastering",
    102: "Plataform diving",
    103: "Playing accordion",
    104: "Playing badminton",
    105: "Playing bagpipes",
    106: "Playing beach volleyball",
    107: "Playing blackjack",
    108: "Playing congas",
    109: "Playing drums",
    110: "Playing field hockey",
    111: "Playing flauta",
    112: "Playing guitarra",
    113: "Playing harmonica",
    114: "Playing ice hockey",
    115: "Playing kickball",
    116: "Playing lacrosse",
    117: "Playing piano",
    118: "Playing polo",
    119: "Playing pool",
    120: "Playing racquetball",
    121: "Playing rubik cube",
    122: "Playing saxophone",
    123: "Playing squash",
    124: "Playing ten pins",
    125: "Playing violin",
    126: "Playing water polo",
    127: "Pole vault",
    128: "Polishing forniture",
    129: "Polishing shoes",
    130: "Powerbocking",
    131: "Preparing pasta",
    132: "Preparing salad",
    133: "Putting in contact lenses",
    134: "Putting on makeup",
    135: "Putting on shoes",
    136: "Rafting",
    137: "Raking leaves",
    138: "Removing curlers",
    139: "Removing ice from car",
    140: "Riding bumper cars",
    141: "River tubing",
    142: "Rock climbing",
    143: "Rock-paper-scissors",
    144: "Rollerblading",
    145: "Roof shingle removal",
    146: "Rope skipping",
    147: "Running a marathon",
    148: "Sailing",
    149: "Scuba diving",
    150: "Sharpening knives",
    151: "Shaving",
    152: "Shaving legs",
    153: "Shot put",
    154: "Shoveling snow",
    155: "Shuffleboard",
    156: "Skateboarding",
    157: "Skiing",
    158: "Slacklining",
    159: "Smoking a cigarette",
    160: "Smoking hookah",
    161: "Snatch",
    162: "Snow tubing",
    163: "Snowboarding",
    164: "Spinning",
    165: "Spread mulch",
    166: "Springboard diving",
    167: "Starting a campfire",
    168: "Sumo",
    169: "Surfing",
    170: "Swimming",
    171: "Swinging at the playground",
    172: "Table soccer",
    173: "Tai chi",
    174: "Tango",
    175: "Tennis serve with ball bouncing",
    176: "Throwing darts",
    177: "Trimming branches or hedges",
    178: "Triple jump",
    179: "Tug of war",
    180: "Tumbling",
    181: "Using parallel bars",
    182: "Using the balance beam",
    183: "Using the monkey bar",
    184: "Using the pommel horse",
    185: "Using the rowing machine",
    186: "Using uneven bars",
    187: "Vacuuming floor",
    188: "Volleyball",
    189: "Wakeboarding",
    190: "Walking the dog",
    191: "Washing dishes",
    192: "Washing face",
    193: "Washing hands",
    194: "Waterskiing",
    195: "Waxing skis",
    196: "Welding",
    197: "Windsurfing",
    198: "Wrapping presents",
    199: "Zumba"
}

def load_val_list(txt_path):
    video_names = []
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_names.append(parts[0])
                labels.append(int(parts[2]))
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
                print(f"Warning: Video {name} not found, using zero vector.")
                vec = np.zeros(768, dtype=np.float32)
            else:
                vec = grp['vectors'][:]
            if aggregate and vec.ndim == 2:
                vec = vec.mean(axis=0)
            features.append(vec.astype(np.float32))
    return np.array(features)

def main():
    video_names, original_labels = load_val_list(VAL_TXT)
    all_classes = np.unique(original_labels)
    print(f"Total classes in val set: {len(all_classes)}")

    np.random.seed(RANDOM_SEED)
    if len(all_classes) <= NUM_RANDOM_CLASSES:
        selected_classes = all_classes
    else:
        selected_classes = np.random.choice(all_classes, NUM_RANDOM_CLASSES, replace=False)
    selected_classes = np.sort(selected_classes)
    print(f"Randomly selected classes: {selected_classes.tolist()}")
    for c in selected_classes:
        name = ACTIVITYNET_CLASS_NAMES.get(c, f"Class {c}")
        print(f"  {c:3d}: {name}")

    domains = ['act_vit', 'syn1', 'syn2', 'syn3']
    for domain in domains:
        print(f"\n{'='*50}")
        print(f"Processing domain: {domain}")
        h5_path = DOMAIN_H5[domain]
        if not os.path.exists(h5_path):
            print(f"H5 file not found: {h5_path}, skipping.")
            continue

        all_features = load_features_from_h5(h5_path, video_names, aggregate=True)

        sampled_feats = []
        sampled_labels = []
        for cls in selected_classes:
            idx = np.where(original_labels == cls)[0]
            if len(idx) == 0:
                print(f"  Class {cls} not found in val set, skip.")
                continue
            if len(idx) > SAMPLES_PER_CLASS:
                chosen = np.random.choice(idx, SAMPLES_PER_CLASS, replace=False)
            else:
                chosen = idx
            sampled_feats.append(all_features[chosen])
            sampled_labels.extend([cls] * len(chosen))

        if len(sampled_feats) == 0:
            print("  No samples collected, skip.")
            continue
        X = np.concatenate(sampled_feats, axis=0)
        y = np.array(sampled_labels)
        print(f"  Collected {X.shape[0]} samples from {len(np.unique(y))} classes.")

        print(f"  Running t‑SNE for {domain}...")
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000, verbose=1)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        unique_y = np.unique(y)
        colors = cm.tab10(np.linspace(0, 1, len(unique_y)))
        for i, cls in enumerate(unique_y):
            idx = np.where(y == cls)[0]
            name = ACTIVITYNET_CLASS_NAMES.get(cls, f"Class {cls}")
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1],
                        c=[colors[i]], label=name,
                        alpha=0.8, s=50, edgecolors='none')

        plt.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False,
                   ncol=2 if len(unique_y) > 8 else 1)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        plt.tight_layout()
        out_file = f"{OUTPUT_PREFIX}_{domain}.png"
        plt.savefig(out_file, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  Plot saved to {out_file}")
        plt.close()

    print("\nAll t‑SNE plots generated successfully.")

if __name__ == '__main__':
    main()