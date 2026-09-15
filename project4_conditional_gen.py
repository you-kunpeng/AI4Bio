# project4_conditional_gen.py
# 阶段四：条件生成——在 project3 的 LSTM 上加"性质条件"，按目标 QED 档位生成分子。
# 条件以特殊 token（<QED0>..<QED4>，按 QED 五分位分档）拼在序列最前面。
#
# 训练：python project4_conditional_gen.py --mode train --epochs 5
# 生成：python project4_conditional_gen.py --mode sample --n 200 --qed-bin 4
# 快速试跑：--mode train --epochs 2 --max-molecules 20000
#
# 说明：路线图里的"靶点条件生成（DiffSBDD 等）"需要 3D 蛋白口袋数据和较大算力，
# 这里用"性质条件生成"演示同一套思想（条件 token -> 生成器）；
# 对接筛选（AutoDock Vina）的安装与用法见 README 阶段四小节。
import argparse
import csv
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA = 'data/train_clean.csv'
VOCAB = 'vocab.pkl'
CKPT = 'project4_cond_lstm.pt'
SAMPLES = 'data/samples_conditional.csv'

QED_TOKENS = ['<QED0>', '<QED1>', '<QED2>', '<QED3>', '<QED4>']  # QED 从低到高 5 档


def qed_bin(mol):
    from rdkit.Chem import QED
    q = QED.qed(mol)
    return min(int(q * 5), 4)  # 0~1 分成 5 档


def load_and_condition(max_molecules):
    """读清洗数据，给每个分子算 QED 档位。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.error')
    with open(DATA, newline='', encoding='utf-8') as f:
        smiles = [row['canonical'] for row in csv.DictReader(f)]
    random.seed(42)
    random.shuffle(smiles)
    if max_molecules:
        smiles = smiles[:max_molecules]
    pairs = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            pairs.append((s, qed_bin(mol)))
    return pairs


class CondSmilesDataset(Dataset):
    """序列 = [条件token] + [SOS] + 字符 + [EOS]。"""

    def __init__(self, pairs, stoi, max_len=120):
        self.data = []
        for s, b in pairs:
            ids = [stoi[QED_TOKENS[b]]] + [stoi['<sos>']] + \
                  [stoi.get(c, stoi['<unk>']) for c in s[:max_len]] + [stoi['<eos>']]
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad):
    maxlen = max(len(seq) for seq in batch)
    x = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    for i, seq in enumerate(batch):
        x[i, :len(seq)] = torch.tensor(seq)
    return x[:, :-1], x[:, 1:]


class CondSmilesLSTM(nn.Module):
    """与 project3 相同结构，只是词表里多出 5 个条件 token。"""

    def __init__(self, vocab_size, embed=128, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        out, state = self.lstm(self.embed(x), state)
        return self.fc(out), state


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(VOCAB, 'rb') as f:
        vocab = pickle.load(f)
    stoi, pad, eos = vocab['stoi'], vocab['PAD'], vocab['EOS']
    for t in QED_TOKENS:  # 把条件 token 扩充进词表
        stoi.setdefault(t, len(stoi))

    pairs = load_and_condition(args.max_molecules)
    print(f'带条件标签的训练分子数: {len(pairs)}, 设备: {device}')

    loader = DataLoader(CondSmilesDataset(pairs, stoi), batch_size=args.batch_size,
                        shuffle=True, collate_fn=lambda b: collate(b, pad))
    model = CondSmilesLSTM(len(stoi)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)
        print(f'epoch {epoch}/{args.epochs}  loss={total / seen:.4f}')
        torch.save({'model': model.state_dict(), 'stoi': stoi}, CKPT)
    print(f'模型已保存: {CKPT}')


@torch.no_grad()
def sample_batch(model, stoi, itos, qed_bin_idx, device, batch=64,
                 max_len=120, temperature=0.8):
    """并行生成 batch 条指定 QED 档位的序列。"""
    cond, sos, eos = stoi[QED_TOKENS[qed_bin_idx]], stoi['<sos>'], stoi['<eos>']
    x = torch.tensor([[cond, sos]] * batch, dtype=torch.long, device=device)
    state = None
    finished = torch.zeros(batch, dtype=torch.bool)
    chars = [[] for _ in range(batch)]
    for _ in range(max_len):
        logits, state = model(x, state)
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)
        for i in range(batch):
            if not finished[i] and nxt[i].item() != eos:
                chars[i].append(itos.get(nxt[i].item(), ''))
        finished |= nxt.eq(eos)
        if finished.all():
            break
        x = nxt.unsqueeze(1)
    return [''.join(c) for c in chars]


def generate(args):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    RDLogger.DisableLog('rdApp.error')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    stoi = ckpt['stoi']
    itos = {i: c for c, i in stoi.items()}
    model = CondSmilesLSTM(len(stoi)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    random.seed(0)
    torch.manual_seed(0)
    valid, attempts, max_attempts = [], 0, max(1024, args.n * 50)
    while len(valid) < args.n and attempts < max_attempts:
        for smi in sample_batch(model, stoi, itos, args.qed_bin, device,
                                batch=64, temperature=args.temperature):
            attempts += 1
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                valid.append((smi, QED.qed(mol)))
            if len(valid) >= args.n:
                break

    with open(SAMPLES, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['SMILES', 'QED', 'target_qed_bin'])
        w.writerows([[s, f'{q:.3f}', args.qed_bin] for s, q in valid])
    if valid:
        mean_q = sum(q for _, q in valid) / len(valid)
        print(f'目标 QED 档位 {args.qed_bin}（约 {args.qed_bin / 5:.1f}+）：'
              f'生成 {len(valid)} 个有效分子（尝试 {attempts} 次），平均 QED={mean_q:.3f}')
    else:
        print('没有生成有效分子，请增加训练量（--epochs / --max-molecules）后重试')
    print(f'结果已保存: {SAMPLES}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'sample'], required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-molecules', type=int, default=0)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--qed-bin', type=int, default=4, choices=range(5),
                    help='0~4，越大要求类药性越高')
    ap.add_argument('--temperature', type=float, default=0.8)
    args = ap.parse_args()
    (train if args.mode == 'train' else generate)(args)


if __name__ == '__main__':
    main()
