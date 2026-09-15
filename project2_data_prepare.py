# project2_data_prepare.py
# 阶段二：下载 MOSES 数据集 -> RDKit 清洗 -> 统计分布 -> 构建词表
#
# 历史坑（重要）：MOSES 仓库的 csv 是 Git LFS 文件，raw.githubusercontent.com
# 返回的只是 100 多字节的"指针文本"（第一行 version https://git-lfs.github.com/spec/v1），
# 导致旧版脚本"清洗后数量: 0、词表大小: 4"。
# 修复：改用 media.githubusercontent.com 直接取 LFS 实体内容，并校验文件头。
#
# 用法：
#   python project2_data_prepare.py                 # 下载全量并清洗（train 约 193 万条，清洗约 10 分钟）
#   python project2_data_prepare.py --sample 50000  # 只清洗前 5 万条（快速试跑）
import argparse
import os
import pickle
import sys

import pandas as pd
import requests

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

FILES = ['train.csv', 'test.csv', 'test_scaffolds.csv']

# 按优先级排列的下载源（均为 LFS 实体或可靠镜像）
def urls_for(name):
    media = f'https://media.githubusercontent.com/media/molecularsets/moses/master/data/{name}'
    return [
        media,
        'https://gh-proxy.com/' + media,
        'https://ghproxy.net/' + media,
    ]

LFS_MARKER = 'version https://git-lfs.github.com/spec/v1'


def looks_like_pointer(path):
    """检查本地文件是不是 LFS 指针文本/过小的无效文件。文件不存在时也返回 True。"""
    if not os.path.exists(path):
        return True
    with open(path, 'rb') as f:
        head = f.read(200)
    return head.startswith(b'version https://git-lfs') or os.path.getsize(path) < 1_000_000


def download(name, out_path):
    for i, url in enumerate(urls_for(name)):
        try:
            print(f'[{name}] 尝试第 {i + 1} 个源: {url}')
            with requests.get(url, timeout=120, stream=True) as r:
                if r.status_code != 200:
                    print(f'  状态码 {r.status_code}，换下一个源')
                    continue
                first = next(r.iter_content(chunk_size=256), b'')
                if first.startswith(LFS_MARKER.encode()):
                    print('  仍是 LFS 指针，换下一个源')
                    continue
                size = int(r.headers.get('Content-Length', 0))
                print(f'  开始下载，总大小 {size / 1e6:.1f} MB')
                with open(out_path, 'wb') as f:
                    f.write(first)
                    done = len(first)
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if size and done % (20 << 20) < (1 << 20):
                            print(f'  进度 {done / 1e6:.0f}/{size / 1e6:.0f} MB')
            if not looks_like_pointer(out_path):
                print(f'  完成: {out_path}')
                return True
        except Exception as e:
            print(f'  失败: {e}')
    return False


def canonical(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.error')  # 关掉无效 SMILES 的报错刷屏
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean(split, sample_n):
    """清洗一个 split：canonical 化、去无效、去重。"""
    path = os.path.join(DATA_DIR, f'{split}.csv')
    out_path = os.path.join(DATA_DIR, f'{split}_clean.csv')
    if looks_like_pointer(path):
        print(f'{path} 是 LFS 指针/空文件，需要重新下载')
        return None
    df = pd.read_csv(path)
    col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    if sample_n:
        df = df.head(sample_n)
    print(f'[{split}] 原始数量: {len(df)}, 开始清洗...')
    df = df[[col]].rename(columns={col: 'canonical'})
    df['canonical'] = df['canonical'].astype(str).map(canonical)
    before = len(df)
    df = df.dropna().drop_duplicates('canonical')
    df.to_csv(out_path, index=False)
    print(f'[{split}] 有效 {len(df)} / {before}（无效+重复 {before - len(df)}），已保存 {out_path}')
    return out_path


def property_stats(train_clean):
    """阶段二产出：分子量 / QED / LogP / SA 分布图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer
    from rdkit.Chem import Crippen, Descriptors, QED

    df = pd.read_csv(train_clean)
    mols = [Chem.MolFromSmiles(s) for s in df['canonical'].head(20000)]
    mols = [m for m in mols if m is not None]
    mw = [Descriptors.MolWt(m) for m in mols]
    logp = [Crippen.MolLogP(m) for m in mols]
    qed = [QED.qed(m) for m in mols]
    sa = [sascorer.calculateScore(m) for m in mols]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, vals, title in zip(
        axes.flat, [mw, logp, qed, sa],
        ['Molecular Weight', 'LogP (Crippen)', 'QED', 'SA Score'],
    ):
        ax.hist(vals, bins=50, color='#4c72b0')
        ax.set_title(f'{title}  (n={len(mols)})')
    fig.tight_layout()
    out = 'project2_property_distributions.png'
    fig.savefig(out, dpi=120)
    print(f'性质分布图已保存: {out}（MW/LogP/QED/SA）')


def build_vocab(train_clean):
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    df = pd.read_csv(train_clean)
    chars = sorted(set(''.join(df['canonical'].tolist())))
    stoi = {c: i + 4 for i, c in enumerate(chars)}
    stoi.update({'<pad>': PAD, '<sos>': SOS, '<eos>': EOS, '<unk>': UNK})
    itos = {i: c for c, i in stoi.items()}
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos,
                     'PAD': PAD, 'SOS': SOS, 'EOS': EOS, 'UNK': UNK}, f)
    print(f'词表大小: {len(stoi)}（含 4 个特殊符号，字符集: {"".join(chars)}）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0,
                    help='每个 split 只清洗前 N 条；0 = 全量')
    args = ap.parse_args()

    # 1. 检查/下载原始数据
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path) or looks_like_pointer(path):
            if not download(name, path):
                print(f'\n自动下载 {name} 失败。请手动下载（注意要用 LFS 实体链接，'
                      f'raw.githubusercontent.com 拿到的只是指针）：\n'
                      f'  {urls_for(name)[0]}\n放到 {os.path.abspath(DATA_DIR)} 后重跑。')
                sys.exit(1)
        else:
            print(f'已存在有效 {path}，跳过下载')

    # 2. 清洗
    train_clean = clean('train', args.sample)
    clean('test', args.sample)
    clean('test_scaffolds', args.sample)
    if train_clean is None:
        sys.exit(1)

    # 3. 统计图 + 词表
    property_stats(train_clean)
    build_vocab(train_clean)
    print('完成：data/*_clean.csv, vocab.pkl, project2_property_distributions.png')


if __name__ == '__main__':
    main()
