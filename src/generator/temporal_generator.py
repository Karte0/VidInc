"""
输入h5 : /{dir_name}/vectors -> (n_frames, 768)
输出h5 : 原始: /{dir_name}/vectors
         合成: /{dir_name}_novel{i}/vectors
txt格式: dir_name,class_id


  1. TemporalGenerator: 生成帧token
     - dom_embed作为独立token
  2. 训练循环：每step处理所有类，每类从cls_iters取固定k_shot个样本
     保证每类样本数充足，G每step都能更新
  3. 训练顺序：先更新F_再更新G
"""

import os, random, argparse, h5py, itertools, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",         required=True)
    p.add_argument("--txts",       nargs="+", required=True)
    p.add_argument("--output",     default="synthetic.h5")
    p.add_argument("--Kn",         type=int,   default=3)
    p.add_argument("--hidden",     type=int,   default=768)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--pre_epochs", type=int,   default=15)
    p.add_argument("--iters",      type=int,   default=10)
    p.add_argument("--bs",         type=int,   default=32)
    p.add_argument("--k_shot",     type=int,   default=32,
                   help="每类每step取的样本数，需>=4")
    p.add_argument("--lr_g",       type=float, default=1e-4)
    p.add_argument("--lr_f",       type=float, default=5e-4)
    p.add_argument("--lam_d",      type=float, default=0.5)
    p.add_argument("--lam_c",      type=float, default=0.01)
    p.add_argument("--lam_ce",     type=float, default=0.1)
    p.add_argument("--ot_reg",     type=float, default=0.5)
    p.add_argument("--alpha",      type=float, default=0.5)
    p.add_argument("--gpu",        type=int,   default=3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save",       default="best3.pth")
    return p.parse_args()


# ──────────────────────────────────────────────
# 进度工具
# ──────────────────────────────────────────────
def progress(current, total, prefix='', suffix='', width=30):
    filled = int(width * current / total)
    bar    = '█' * filled + '░' * (width - filled)
    print(f'\r{prefix} |{bar}| {current}/{total} {suffix}',
          end='', flush=True)
    if current == total:
        print()

def fmt_time(s):
    s = int(s)
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    if h > 0: return f"{h}h{m:02d}m{s:02d}s"
    if m > 0: return f"{m}m{s:02d}s"
    return f"{s}s"


# ──────────────────────────────────────────────
# 数据
# ──────────────────────────────────────────────
def load_txt(txt_paths):
    m = {}
    for path in txt_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) < 2: continue
                #m[parts[0].strip()] = int(parts[1].strip())
                m[parts[0].strip()] = int(parts[2].strip())
    print(f"[Labels] {len(m)} 条，来自 {len(txt_paths)} 个txt")
    return m


class FeatureDataset(Dataset):
    def __init__(self, h5_path, txt_paths):
        label_map      = load_txt(txt_paths)
        self.data      = []   # (video_mean, label)
        self.dir_names = []
        self.raw_vecs  = []   # 原始帧序列

        skip = 0
        with h5py.File(h5_path, 'r') as f:
            keys  = list(f.keys())
            total = len(keys)
            print(f"[H5] 共 {total} 个条目，加载中...")
            for i, dn in enumerate(keys):
                progress(i+1, total, prefix='  读取H5')
                if dn not in label_map:
                    skip += 1; continue
                vecs = f[dn]['vectors'][:]       # (T, 768)
                mean = vecs.mean(0)              # (768,)
                self.data.append((mean.astype(np.float32), label_map[dn]))
                self.dir_names.append(dn)
                self.raw_vecs.append(vecs.astype(np.float32))

        if skip:
            print(f"[警告] {skip} 个dir_name无标签，已跳过")

        cls_sorted = sorted(set(l for _, l in self.data))
        remap      = {c: i for i, c in enumerate(cls_sorted)}
        self.data  = [(f, remap[l]) for f, l in self.data]

        self.feat_dim    = self.data[0][0].shape[0]
        self.num_classes = len(cls_sorted)
        print(f"[Dataset] {len(self.data)}样本 | "
              f"dim={self.feat_dim} | {self.num_classes}类")

        self.class_indices = defaultdict(list)
        for i, (_, l) in enumerate(self.data):
            self.class_indices[l].append(i)

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        raw_vecs  = self.raw_vecs[i]
        norms     = np.linalg.norm(raw_vecs, axis=-1, keepdims=True) + 1e-8
        vecs_norm = (raw_vecs / norms).astype(np.float32)
        _, l      = self.data[i]
        return torch.tensor(vecs_norm), torch.tensor(l)


def inf(loader):
    while True:
        for b in loader: yield b


# ──────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────
class TemporalGenerator(nn.Module):
    """
    时序感知生成器（条件加法融合版）

    条件向量直接加到帧embedding，不使用独立条件token：
      - proto_vec : 视频均值投影，捕获样本级子模式信息
      - dom_vec   : 域条件
      - cls_vec   : 类别条件
    三者广播后与帧tokens相加，Transformer只处理T帧
    输出直接取全部T帧，无需跳过offset
    """
    def __init__(self, feat_dim, num_classes, Kn,
                 hidden=512, n_layers=4, n_heads=8, max_frames=25):
        super().__init__()
        self.total_dom = 1 + Kn

        # 帧特征投影
        self.frame_proj = nn.Linear(feat_dim, hidden)
        self.dom_embed = nn.Embedding(self.total_dom, hidden)

        with torch.no_grad():
            W = torch.zeros(self.total_dom, hidden)
            nn.init.orthogonal_(W)          # 正交矩阵，行向量两两正交
            W = W * (hidden ** 0.5)         # 缩放到合适幅度
            self.dom_embed.weight.copy_(W)

        # Transformer Encoder (Pre-LN，训练更稳定)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden,
            dropout=0.0,
            batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # 输出delta，初始化为0保证训练初期稳定
        self.out_proj = nn.Linear(hidden, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, frames, cls, dom):
        """
        frames : (B, T, 768)
        cls    : (B,)
        dom    : (B,)
        return : (B, T, 768) frames + delta
        """
        B, T, _ = frames.shape

        frame_tokens = self.frame_proj(frames)                    # (B, T, hidden)

        # 域/类别条件：广播到每一帧
        dom_tok = self.dom_embed(dom).unsqueeze(1)                # (B, 1, hidden)

        # 条件直接加到每一帧（广播，不拼接token）
        tokens = torch.cat([dom_tok, frame_tokens], dim=1)           # (B, T+1, hidden)

        # Transformer：输入输出均为T帧
        out = self.transformer(tokens)                            # (B, T, hidden)

        # 直接取全部帧输出，无需跳过任何offset
        delta = self.out_proj(out[:, 1:, :])
        return frames + delta



class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden, bias=False), nn.LayerNorm(hidden,bias=False), nn.GELU(),
            nn.Linear(hidden, out_d, bias=False))
    def forward(self, x): return self.net(x)


# ──────────────────────────────────────────────
# OT
# ──────────────────────────────────────────────
def sinkhorn_stable(a, b, C, reg=0.5, n_iter=8):
    log_a = torch.log(a.clamp(min=1e-8))
    log_b = torch.log(b.clamp(min=1e-8))
    C     = C.clamp(min=0, max=2)
    logK  = -C / reg
    f     = torch.zeros_like(a)
    g     = torch.zeros_like(b)
    for _ in range(n_iter):
        f = log_a - torch.logsumexp(logK + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(logK + f.unsqueeze(1), dim=0)
        f = f.clamp(-10, 10)
        g = g.clamp(-10, 10)
    log_T = logK + f.unsqueeze(1) + g.unsqueeze(0)
    T     = log_T.clamp(max=0).exp()
    return (T * C).sum()

def ot_dist(x, y, reg):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    C = (1 - x @ y.t()).clamp(min=0, max=2)
    n, m = len(x), len(y)
    a = x.new_ones(n) / n
    b = y.new_ones(m) / m
    dist = sinkhorn_stable(a, b, C, reg)
    if torch.isnan(dist) or torch.isinf(dist):
        return torch.tensor(0., device=x.device, requires_grad=True)
    return dist

def ged(xa, xa2, xb, xb2, reg):
    d = 2*ot_dist(xa,xb,reg) - ot_dist(xa,xa2,reg) - ot_dist(xb,xb2,reg)
    return d.clamp(min=-10, max=100)


# ──────────────────────────────────────────────
# 预训练
# ──────────────────────────────────────────────
def pretrain_clf(model, dataset, device, epochs, bs, lr):
    loader = DataLoader(dataset, bs*2, shuffle=True,
                        num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot, cor, n = 0., 0, 0
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            feats = frames.mean(dim=1)                            # (B, 768)
            loss  = F.cross_entropy(model(feats), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(frames)
            cor += (model(feats).argmax(1) == labels).sum().item()
            n   += len(frames)
        elapsed = time.time() - t0
        eta     = elapsed/(ep+1) * (epochs-ep-1)
        progress(ep+1, epochs, prefix='  预训练',
                 suffix=f'loss={tot/n:.4f} acc={cor/n*100:.1f}% '
                        f'已用{fmt_time(elapsed)} ETA {fmt_time(eta)}')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────
def train(args):
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    dev = torch.device(f"cuda:{args.gpu}"
                       if torch.cuda.is_available() else "cpu")
    print(f"[Device] {dev}")

    ds        = FeatureDataset(args.h5, args.txts)
    Kn        = args.Kn
    D         = ds.feat_dim
    C         = ds.num_classes
    novel_ids = list(range(1, 1+Kn))

    G  = TemporalGenerator(D, C, Kn,
                           args.hidden, args.n_layers, args.n_heads).to(dev)
    F_ = MLP(D, C, 128).to(dev)
    Yp = MLP(D, C, 128).to(dev)

    print("\n[预训练语义分类器]")
    Yp = pretrain_clf(Yp, ds, dev, args.pre_epochs, args.bs, args.lr_f)

    opt_G = torch.optim.Adam(G.parameters(),  args.lr_g, weight_decay=1e-5)
    opt_F = torch.optim.Adam(F_.parameters(), args.lr_f, weight_decay=1e-5)

    # full_it：用于更新F_
    full_it = inf(DataLoader(ds, args.bs, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True))

    # cls_iters：每类单独的无限迭代器
    k = args.k_shot
    cls_iters = {
        c: inf(DataLoader(
            Subset(ds, idxs),
            batch_size=k,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True))
        for c, idxs in ds.class_indices.items()
        if len(idxs) >= k
    }
    valid_cls = sorted(cls_iters.keys())
    print(f"[ClassIter] 有效类别: {len(valid_cls)} / {C} "
          f"(每类取 k_shot={k} 个样本)")

    best      = float('inf')
    t_train   = time.time()
    nan_count = 0

    print(f"\n[训练开始] {args.epochs}ep × {args.iters}iter | "
          f"Kn={Kn} | C={C} | 每step处理{len(valid_cls)}类\n")

    for ep in range(args.epochs):
        G.train(); F_.train()
        logs  = defaultdict(float)
        t_ep  = time.time()

        for step in range(args.iters):
            random.shuffle(novel_ids)

            # ── 更新 F_ ──────────────────────────────────────────
            opt_F.zero_grad()
            frames_f, y_f = next(full_it)
            frames_f, y_f = frames_f.to(dev), y_f.to(dev)
            B_f           = frames_f.shape[0]
            x_mean_f      = frames_f.mean(dim=1)                  # (B, 768)

            l_f = (1 - args.alpha) * F.cross_entropy(F_(x_mean_f), y_f)
            for nid in novel_ids:
                nl = torch.full((B_f,), nid, dtype=torch.long, device=dev)
                with torch.no_grad():
                    xn = G(frames_f, y_f, nl)                     # (B, T, 768)
                l_f = l_f + (args.alpha / Kn) * F.cross_entropy(
                    F_(xn.mean(dim=1)), y_f)
                del xn

            if not (torch.isnan(l_f) or torch.isinf(l_f)):
                l_f.backward()
                nn.utils.clip_grad_norm_(F_.parameters(), 1.0)
                opt_F.step()

            # ── 更新 G：每类单独backward，梯度累积 ───────────────
            opt_G.zero_grad()
            n_valid       = 0
            has_valid     = False
            # 仅用于logging
            l_nov_log = l_cyc_log = 0.0
            l_g_log   = 0.0

            for c in valid_cls:
                frames, y = next(cls_iters[c])
                frames, y = frames.to(dev), y.to(dev)
                B         = frames.shape[0]
                if B < 4: continue

                x_mean  = frames.mean(dim=1)                      # (B, 768)
                n       = B // 2
                cls_t   = torch.full((B,), c,   dtype=torch.long, device=dev)
                src_dom = torch.zeros(B,         dtype=torch.long, device=dev)

                l_nov = l_div = l_cyc = l_ce = torch.tensor(0., device=dev)
                xn_means = []

                for nid in novel_ids:
                    nl  = torch.full((B,), nid, dtype=torch.long, device=dev)
                    xn  = G(frames, cls_t, nl)                    # (B, T, 768)
                    xn_mean = xn.mean(dim=1)                      # (B, 768)
                    xn_means.append(xn_mean)

                    # l_nov：当场计算
                    if n >= 2:
                        l_nov = l_nov + ged(
                            xn_mean[:n], xn_mean[n:2*n],
                            x_mean[:n],  x_mean[n:2*n],
                            args.ot_reg)

                    # l_ce：当场计算
                    l_ce = l_ce + F.cross_entropy(Yp(xn_mean), y)

                    # l_cyc：detach截断，避免嵌套计算图
                    l_cyc = l_cyc + F.l1_loss(
                        G(xn.detach(), cls_t, src_dom), frames)

                    del xn  # 立即释放，只保留xn_mean

                # l_div：只用xn_means
                if n >= 2:
                    for i, j in itertools.combinations(range(Kn), 2):
                        l_div = l_div + ged(
                            xn_means[i][:n], xn_means[i][n:2*n],
                            xn_means[j][:n], xn_means[j][n:2*n],
                            args.ot_reg)

                # ★ 每类单独算loss，除以总类数归一化，当场backward
                l_g_c = (-args.lam_d  * (l_nov + l_div)
                         + args.lam_c *  l_cyc
                         + args.lam_ce * l_ce) / len(valid_cls)

                if not (torch.isnan(l_g_c) or torch.isinf(l_g_c)):
                    l_g_c.backward()          # ★ 当场释放本类计算图
                    has_valid  = True
                    n_valid   += 1
                    l_g_log   += l_g_c.item() * len(valid_cls)    # 还原未归一化值供log
                    l_nov_log += l_nov.item()
                    l_cyc_log += l_cyc.item()
                else:
                    nan_count += 1
                    if nan_count % 10 == 1:
                        print(f"\n  [警告] step {step} 类{c} G loss NaN，跳过"
                              f"(累计{nan_count}次)")

                # ★ 显式释放本类所有tensor，不等下一类
                del frames, y, x_mean, xn_means
                del l_nov, l_div, l_cyc, l_ce, l_g_c

            if has_valid:
                nan_count = 0
                nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()
                l_g_display = l_g_log / len(valid_cls)
            else:
                l_g_display = 0.0

            logs['g']   += l_g_display
            logs['f']   += l_f.item() if not torch.isnan(l_f) else 0
            logs['nov'] += l_nov_log
            logs['cyc'] += l_cyc_log

            elapsed_ep = time.time() - t_ep
            eta_ep     = elapsed_ep / (step + 1) * (args.iters - step - 1)
            progress(step + 1, args.iters,
                     prefix=f'  Ep{ep+1:03d}',
                     suffix=f'G={l_g_display:+.3f} '
                            f'F={l_f.item():.3f} '
                            f'nov={l_nov_log:.3f} '
                            f'| ETA {fmt_time(eta_ep)}')

        n             = args.iters
        ep_time       = time.time() - t_ep
        total_elapsed = time.time() - t_train
        eta_total     = total_elapsed / (ep + 1) * (args.epochs - ep - 1)
        saved_mark    = ''

        if logs['f'] < best and logs['f'] > 0:
            best = logs['f']
            torch.save({'G': G.state_dict(), 'F': F_.state_dict(),
                        'C': C, 'Kn': Kn, 'D': D}, args.save)
            saved_mark = ' ✓'

        print(f"\nEp{ep+1:03d}/{args.epochs} | "
              f"G={logs['g']/n:+.3f} F={logs['f']/n:.3f} "
              f"nov={logs['nov']/n:.3f} cyc={logs['cyc']/n:.3f} "
              f"| ep:{fmt_time(ep_time)} 总ETA:{fmt_time(eta_total)}"
              f"{saved_mark}")

    print(f"\n[训练完成] 总耗时:{fmt_time(time.time()-t_train)} "
          f"最优F={best:.4f} -> {args.save}")
    return G, ds, dev




@torch.no_grad()
def synthesize(G, dataset, Kn, device, output_h5_prefix):
    """
    output_h5_prefix: 如 'output/novel'
    会生成 output/novel_1.h5, output/novel_2.h5, ..., output/novel_Kn.h5
    不保留原始数据
    """
    G.eval()
    total = len(dataset.dir_names)

    # 预先打开所有h5文件
    h5_files = {}
    for nid in range(1, 1 + Kn):
        path = f"{output_h5_prefix}_{nid}.h5"
        h5_files[nid] = h5py.File(path, 'w')
        print(f"[合成] 域{nid} -> {path}")

    t0 = time.time()
    try:
        for idx, dn in enumerate(dataset.dir_names):
            _, label  = dataset.data[idx]
            raw_vecs  = dataset.raw_vecs[idx]                      # (T, 768)

            norms     = np.linalg.norm(raw_vecs, axis=-1,
                                       keepdims=True) + 1e-8
            vecs_norm = (raw_vecs / norms).astype(np.float32)

            x   = torch.tensor(vecs_norm).unsqueeze(0).to(device)  # (1,T,768)
            cls = torch.tensor([label], dtype=torch.long, device=device)

            for nid in range(1, 1 + Kn):
                nl  = torch.tensor([nid], dtype=torch.long, device=device)
                out = G(x, cls, nl).squeeze(0).cpu().numpy()        # (T,768)

                # 只存合成数据，不存原始
                h5_files[nid].require_group(dn).create_dataset(
                    'vectors', data=out.astype(np.float32), dtype='f')

            elapsed = time.time() - t0
            eta     = elapsed / (idx + 1) * (total - idx - 1)
            progress(idx + 1, total, prefix='  合成',
                     suffix=f'已用{fmt_time(elapsed)} ETA {fmt_time(eta)}')

    finally:
        # 无论是否出错，都关闭所有文件
        for nid, f in h5_files.items():
            f.close()

    print(f"\n[合成完成] {total}条 × {Kn}个域 = {total*Kn}条合成数据"
          f" | 耗时{fmt_time(time.time()-t0)}")




if __name__ == '__main__':
    args = get_args()
    G, ds, dev = train(args)
    synthesize(G, ds, args.Kn, dev, args.output)

