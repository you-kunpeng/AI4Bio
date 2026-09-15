# project3_smiles_lstm.py
# 阶段三：字符级 LSTM 生成 SMILES（最经典的分子生成入门模型）
#
# 训练：python project3_smiles_lstm.py --mode train --epochs 5
# 采样：python project3_smiles_lstm.py --mode sample --n 1000
# 快速试跑：python project3_smiles_lstm.py --mode train --epochs 2 --max-molecules 20000
#           python project3_smiles_lstm.py --mode sample --n 200
import argparse
import csv
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA = 'data/train_clean.csv'
VOCAB = 'vocab.pkl'
CKPT = 'project3_lstm.pt'
SAMPLES = 'data/samples_lstm.csv'


# ---------- 数据 ----------
class SmilesDataset(Dataset):
    """每条样本 = [SOS] + 字符序列 + [EOS]，用 padding 对齐成 batch。"""

    def __init__(self, smiles_list, stoi, max_len=120):
        self.data = []
        for s in smiles_list:
            ids = [stoi['<sos>']] + [stoi.get(c, stoi['<unk>']) for c in s[:max_len]] + [stoi['<eos>']]
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad):
    """把不等长序列 pad 到 batch 内最长，返回 (x, y)：y 是 x 右移一位的目标。"""
    maxlen = max(len(seq) for seq in batch)
    x = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    for i, seq in enumerate(batch):
        x[i, :len(seq)] = torch.tensor(seq)
    return x[:, :-1], x[:, 1:]  # 输入去掉最后一个，目标去掉第一个


# ---------- 模型 ----------
class SmilesLSTM(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        emb = self.embed(x)
        out, state = self.lstm(emb, state)
        return self.fc(out), state


# ---------- 训练 ----------
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(VOCAB, 'rb') as f:
        vocab = pickle.load(f)
    stoi, pad, eos = vocab['stoi'], vocab['PAD'], vocab['EOS']

    with open(DATA, newline='', encoding='utf-8') as f:
        smiles = [row['canonical'] for row in csv.DictReader(f)]
    random.seed(42)
    random.shuffle(smiles)
    if args.max_molecules:
        smiles = smiles[:args.max_molecules]
    print(f'训练分子数: {len(smiles)}, 设备: {device}')

    loader = DataLoader(SmilesDataset(smiles, stoi), batch_size=args.batch_size,
                        shuffle=True, collate_fn=lambda b: collate(b, pad))
    model = SmilesLSTM(len(stoi)).to(device)
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
        torch.save({'model': model.state_dict(), 'vocab_size': len(stoi)}, CKPT)
    print(f'模型已保存: {CKPT}')


# ---------- 采样 ----------
@torch.no_grad()
def sample_batch(model, stoi, itos, device, batch=64, max_len=120, temperature=0.8):
    """一次并行生成 batch 条序列（CPU/GPU 都比逐条快几十倍）。"""
    sos, eos = stoi['<sos>'], stoi['<eos>']
    x = torch.full((batch, 1), sos, dtype=torch.long, device=device)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(VOCAB, 'rb') as f:
        vocab = pickle.load(f)
    itos, stoi = vocab['itos'], vocab['stoi']

    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = SmilesLSTM(ckpt['vocab_size']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    random.seed(0)
    torch.manual_seed(0)
    out = []
    while len(out) < args.n:
        out += [s for s in sample_batch(model, stoi, itos, device,
                                        batch=64, temperature=args.temperature) if s]
    out = out[:args.n]

    with open(SAMPLES, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['SMILES'])
        w.writerows([[s] for s in out])
    print(f'已生成 {len(out)} 个分子 -> {SAMPLES}（有效/唯一性请用 project5_evaluation.py 评估）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'sample'], required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-molecules', type=int, default=0, help='0 = 用全部清洗后数据')
    ap.add_argument('--n', type=int, default=1000, help='采样分子数')
    ap.add_argument('--temperature', type=float, default=0.8)
    args = ap.parse_args()
    (train if args.mode == 'train' else generate)(args)


if __name__ == '__main__':
    main()
