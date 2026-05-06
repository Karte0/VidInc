
import matplotlib.pyplot as plt
import numpy as np

act_ucf = [0.355886, 0.276157, 0.408243, 0.396991, 0.310110, 0.277180,
           0.331985, 0.431735, 0.412858, 0.365699, 0.312023, 0.304518,
           0.435890, 0.235904, 0.293832, 0.399897, 0.400421, 0.439837,
           0.470766, 0.417670, 0.272448, 0.451110, 0.334954, 0.271300,
           0.246131, 0.277569, 0.335819, 0.420993, 0.290652, 0.405486,
           0.352506, 0.362821, 0.399885, 0.420211, 0.323719, 0.436893,
           0.338701, 0.275775, 0.467817, 0.473070, 0.346859, 0.518081,
           0.496854]

act_fcv = [0.176942, 0.198281, 0.189804, 0.179700, 0.143866, 0.227485,
           0.191542, 0.191253, 0.207659, 0.177352, 0.118000, 0.152899,
           0.261781, 0.169023, 0.163771, 0.237337, 0.229411, 0.275922,
           0.297032, 0.235285, 0.144924, 0.227106, 0.206389, 0.302163,
           0.214158, 0.140281, 0.347270, 0.283236, 0.300160, 0.193184,
           0.186450, 0.235780, 0.137110, 0.175036, 0.185931, 0.212157,
           0.132985, 0.299273, 0.306372, 0.147157, 0.275359, 0.212132,
           0.241735]

ucf_fcv = [0.314207, 0.218564, 0.319433, 0.406305, 0.289955, 0.256916,
           0.279617, 0.378305, 0.471564, 0.325341, 0.340139, 0.255274,
           0.423978, 0.197212, 0.279301, 0.294356, 0.392934, 0.410642,
           0.348998, 0.438400, 0.261398, 0.383948, 0.254271, 0.346070,
           0.256953, 0.284529, 0.452732, 0.451768, 0.365875, 0.407761,
           0.372172, 0.354938, 0.371561, 0.384728, 0.398743, 0.429499,
           0.337737, 0.382190, 0.419607, 0.413179, 0.337374, 0.577517,
           0.534909]

num_classes = 43
domain_pairs = ['ActivityNet vs UCF101', 'ActivityNet vs FCVID', 'UCF101 vs FCVID']
colors = ['#AED6F1', '#FAD7A1', '#A9DFBF']

max_per_class = [max(act_ucf[i], act_fcv[i], ucf_fcv[i]) for i in range(num_classes)]
sorted_max_indices = np.argsort(max_per_class)[::-1]
max1_idx = sorted_max_indices[0]
max2_idx = sorted_max_indices[1]

min_per_class = [min(act_ucf[i], act_fcv[i], ucf_fcv[i]) for i in range(num_classes)]
sorted_min_indices = np.argsort(min_per_class)
min_idx = sorted_min_indices[0]


values_max1 = [act_ucf[max1_idx], act_fcv[max1_idx], ucf_fcv[max1_idx]]
values_max2 = [act_ucf[max2_idx], act_fcv[max2_idx], ucf_fcv[max2_idx]]
values_min  = [act_ucf[min_idx],  act_fcv[min_idx],  ucf_fcv[min_idx]]

mean_act_ucf = np.mean(act_ucf)
mean_act_fcv = np.mean(act_fcv)
mean_ucf_fcv = np.mean(ucf_fcv)
values_mean = [mean_act_ucf, mean_act_fcv, mean_ucf_fcv]

values_overall = [0.19, 0.10, 0.20]

labels = [
    f'Class {max1_idx}',
    f'Class {max2_idx}',
    '...',
    'Mean',
    '...',
    f'Class {min_idx}',
    'Overall'
]

data_matrix = [
    values_max1,
    values_max2,
    [0, 0, 0],
    values_mean,
    [0, 0, 0],
    values_min,
    values_overall
]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 8))

for i, (pair_name, color) in enumerate(zip(domain_pairs, colors)):
    heights = [data_matrix[j][i] for j in range(len(labels))]
    offset = width * i
    rects = ax.bar(x + offset, heights, width, label=pair_name, color=color, edgecolor='white', linewidth=0.5)

max_height = max(max(v) for v in data_matrix if max(v) > 0)
for i in range(len(labels)):
    for j in range(3):
        v = data_matrix[i][j]
        if v > 0:
            ax.text(x[i] + j * width, v + max_height * 0.015, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=18)

ax.text(x[2] + width, max_height * 0.5, '...', ha='center', va='center', fontsize=24, color='gray')
ax.text(x[4] + width, max_height * 0.5, '...', ha='center', va='center', fontsize=24, color='gray')

ax.set_ylim(0, max_height * 1.15)
ax.set_ylabel('MMD', fontsize=24)
ax.set_xlabel('Category', fontsize=24)
ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=18)
ax.tick_params(axis='y', labelsize=18)

ax.legend(loc='upper right', framealpha=0.9, fontsize=32, prop={'size': 32})

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('top_mmd_sorted.png', dpi=200, bbox_inches='tight')
plt.show()