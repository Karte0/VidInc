import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 21,
    'axes.titlesize': 24,
    'axes.labelsize': 22.5,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'figure.titlesize': 27,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.format': 'png',
})

log_files = {
    "fine_tune": "log_transformer_acc_bwt_fine_tune_1520100.txt",
    "ewc": "log_transformer_acc_bwt_ewc_1520100.txt",
    "si": "log_transformer_acc_bwt_si_1520100.txt",
    "er": "log_transformer_acc_bwt_er_1520100.txt",
    "gdumb": "log_transformer_acc_bwt_gdumb_1520100.txt",
    "derpp": "log_transformer_acc_bwt_derpp_1520100.txt",
    "joint": "log_transformer_acc_bwt_joint_1520100.txt"
}

results = {}

for method, filename in log_files.items():
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found, skipping.")
        continue

    sections = re.split(r'(Evaluating \w+ \(method=.*?\)\n)', content)
    for i in range(1, len(sections), 2):
        header = sections[i]
        body = sections[i+1] if i+1 < len(sections) else ""
        ds_match = re.search(r'Evaluating (\w+) \(method=', header)
        if not ds_match:
            continue
        ds = ds_match.group(1)

        task_blocks = re.split(r'--- Task \d+ ---', body)
        task1_accs = []
        for block in task_blocks[1:]:
            match = re.search(r'Accuracy on Task 1 val set:\s*([\d.]+)', block)
            if match:
                task1_accs.append(float(match.group(1)))
        if len(task1_accs) == 10:
            results.setdefault(ds, {})[method] = task1_accs
        else:
            print(f"Warning: {method} on {ds} has {len(task1_accs)} task1 entries (expected 10).")

print("=== Data Parsing Summary ===")
for ds in results:
    print(f"{ds}: methods = {list(results[ds].keys())}")
print("============================\n")

colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
method_styles = {
    "fine_tune": ("-", "o", "Fine-tuning"),
    "ewc": ("--", "s", "EWC"),
    "si": (":", "^", "SI"),
    "er": ("-.", "D", "ER"),
    "gdumb": ("-", "v", "GDumb"),
    "derpp": ("--", "p", "DER++"),
    "joint": (":", "*", "Joint")
}
dataset_fullname = {"actnet": "ActivityNet", "fcvid": "FCVID", "ucf": "UCF101"}

for ds, methods_data in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    has_any = False
    for i, (method, style) in enumerate(method_styles.items()):
        if method in methods_data:
            accs = methods_data[method]
            tasks = list(range(1, 11))
            ax.plot(tasks, accs, linestyle=style[0], marker=style[1],
                    color=colors[i], label=style[2], markersize=8, linewidth=2.5)
            has_any = True
    if not has_any:
        print(f"No methods with data for {ds}, skipping plot.")
        plt.close()
        continue

    ax.set_xlabel('Task Number')
    ax.set_ylabel('ACC on Task 1')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)

    if ds in ['fcvid', 'ucf']:
        ax.legend(loc='lower left')
    else:
        ax.legend(loc='best')

    ax.grid(True, linestyle='--', alpha=0.7)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.95)
    fig.tight_layout(pad=0.3)

    plt.savefig(f'{ds}_task1_forgetting_curves.png', dpi=600, bbox_inches='tight')
    plt.show()

print("All plots generated successfully.")