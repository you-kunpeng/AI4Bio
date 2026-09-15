# project6_smiles_vae.py
# 阶段六：复现一个简化版 SMILES VAE（ChemVAE / GVAE 的最小可运行版本）。
# 学到的东西：潜在空间、重参数化技巧、KL 散度、分子插值、性质预测头。
#
# 训练：     python project6_smiles_vae.py --mode train --epochs 5
# 分子插值： python project6_smiles_vae.py --mode interpolate --n 5
# 快速试跑： python project6_smiles_vae.py --mode train --epochs 2 --max-molecules 20000
#
# 结构：encoder(LSTM) -> mu, logvar -> z (重参数化) -> decoder(LSTM, 以 z 为初始隐状态)
#       额外一个小回归头从 z 预测 LogP，便于后面做"在潜在空间里找高 LogP 分子"。
import argparse
import csv
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA = 'data/train_clean.csv'
VOCAB = 'vocab.pkl'
CKPT = 'project6_vae.pt'


class VaeDataset(Dataset):
    def __init__(self, smiles_list, stoi, max_len=80):
        from rdkit import Chem
        from rdkit.Chem import Crippen
        self.items = []
        for s in smiles_list:
            ids = [stoi['<sos>']] + [stoi.get(c, stoi['<unk>']) for c in s[:max_len]] + [stoi['<eos>']]
            mol = Chem.MolFromSmiles(s)
            logp = Crippen.MolLogP(mol) if mol is not None else 0.0
            self.items.append((ids, logp))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad):
    maxlen = max(len(seq) for seq, _ in batch)
    x = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    logps = torch.zeros(len(batch))
    for i, (seq, logp) in enumerate(batch):
        x[i, :len(seq)] = torch.tensor(seq)
        logps[i] = logp
    return x[:, :-1], x[:, 1:], logps


class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, latent=64):
        super().__init__()
        self.latent = latent
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, embed)
        self.encoder = nn.LSTM(embed, hidden, batch_first=True)
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.z_to_state = nn.Linear(latent, 2 * hidden)  # 生成 decoder 的 (h, c)
        self.decoder = nn.LSTM(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.logp_head = nn.Linear(latent, 1)  # 从 z 回归 LogP

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)  # 重参数化技巧

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.encoder(emb)
        mu, logvar = self.to_mu(h[-1]), self.to_logvar(h[-1])
        z = self.reparameterize(mu, logvar) if self.training else mu
        hc = self.z_to_state(z)
        h0, c0 = hc[:, :self.hidden].unsqueeze(0), hc[:, self.hidden:].unsqueeze(0)
        # decoder 输入 = x 本身（teacher forcing），由 z 决定初始状态
        dout, _ = self.decoder(emb, (h0.contiguous(), c0.contiguous()))
        logits = self.out(dout)
        logp_pred = self.logp_head(z).squeeze(-1)
        return logits, mu, logvar, logp_pred


def vae_loss(logits, y, mu, logvar, logp_pred, logp_true, pad, beta=0.5):
    recon = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
    # KL( N(mu, var) || N(0,1) )，对 batch 求平均
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    prop = nn.functional.mse_loss(logp_pred, logp_true)
    return recon + beta * kl + prop, recon.item(), kl.item(), prop.item()


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(VOCAB, 'rb') as f:
        vocab = pickle.load(f)
    stoi, pad = vocab['stoi'], vocab['PAD']

    with open(DATA, newline='', encoding='utf-8') as f:
        smiles = [row['canonical'] for row in csv.DictReader(f)]
    random.seed(42)
    random.shuffle(smiles)
    if args.max_molecules:
        smiles = smiles[:args.max_molecules]
    print(f'训练分子数: {len(smiles)}, 设备: {device}')

    loader = DataLoader(VaeDataset(smiles, stoi), batch_size=args.batch_size,
                        shuffle=True, collate_fn=lambda b: collate(b, pad))
    model = SmilesVAE(len(stoi)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {'loss': 0.0, 'recon': 0.0, 'kl': 0.0, 'logp': 0.0}
        n = 0
        for x, y, logp in loader:
            x, y, logp = x.to(device), y.to(device), logp.to(device)
            logits, mu, logvar, logp_pred = model(x)
            loss, recon, kl, prop = vae_loss(logits, y, mu, logvar, logp_pred, logp, pad)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot['loss'] += loss.item(); tot['recon'] += recon
            tot['kl'] += kl; tot['logp'] += prop; n += 1
        print(f"epoch {epoch}/{args.epochs}  loss={tot['loss'] / n:.4f}  "
              f"recon={tot['recon'] / n:.4f}  kl={tot['kl'] / n:.4f}  "
              f"logp_mse={tot['logp'] / n:.4f}")
        torch.save({'model': model.state_dict(), 'vocab_size': len(stoi)}, CKPT)
    print(f'模型已保存: {CKPT}')


@torch.no_grad()
def interpolate(args):
    """在潜在空间里对两个真实分子做线性插值，观察中间分子如何渐变。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.error')
    device = 'cpu'
    with open(VOCAB, 'rb') as f:
        vocab = pickle.load(f)
    stoi, pad, sos, eos = vocab['stoi'], vocab['PAD'], vocab['SOS'], vocab['EOS']
    itos = {i: c for c, i in stoi.items()}

    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = SmilesVAE(ckpt['vocab_size']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with open(DATA, newline='', encoding='utf-8') as f:
        smiles = [row['canonical'] for row in csv.DictReader(f)][:500]
    mols = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    a, b = mols[0], mols[1]
    print(f'插值端点 A: {a}\n插值端点 B: {b}\n')

    def encode(s):
        ids = [sos] + [stoi.get(c, stoi['<unk>']) for c in s]
        x = torch.tensor([ids])
        _, (h, _) = model.encoder(model.embed(x))
        return model.to_mu(h[-1])

    za, zb = encode(a), encode(b)
    results = []
    for t in torch.linspace(0, 1, args.n + 2).tolist():
        z = za * (1 - t) + zb * t
        hc = model.z_to_state(z).unsqueeze(0)
        h0 = hc[:, :, :model.hidden].contiguous()
        c0 = hc[:, :, model.hidden:].contiguous()
        x = torch.tensor([[sos]])
        chars = []
        state = (h0, c0)
        for _ in range(100):
            dout, state = model.decoder(model.embed(x), state)
            logits = model.out(dout)  # decoder 输出是隐状态，需再过输出层映射回词表
            nxt = logits[0, -1].argmax().item()  # 插值时用贪心，结果更稳定
            if nxt == eos:
                break
            chars.append(itos.get(nxt, ''))
            x = torch.tensor([[nxt]])
        smi = ''.join(chars)
        ok = Chem.MolFromSmiles(smi) is not None
        results.append(smi)
        print(f't={t:.2f}  {"有效" if ok else "无效"}  {smi}')

    with open('data/vae_interpolation.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'SMILES'])
        for i, smi in enumerate(results):
            w.writerow([i / (len(results) - 1), smi])
    print('已保存: data/vae_interpolation.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'interpolate'], required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-molecules', type=int, default=0)
    ap.add_argument('--n', type=int, default=5, help='插值中间点数')
    args = ap.parse_args()
    (train if args.mode == 'train' else interpolate)(args)


if __name__ == '__main__':
    main()
