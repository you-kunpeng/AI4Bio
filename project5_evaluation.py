# project5_evaluation.py
# 阶段五：评价体系——输入一批生成的 SMILES，输出完整指标报告 + 分布图。
#
# 用法：
#   python project5_evaluation.py --generated data/samples_lstm.csv
#   python project5_evaluation.py --generated data/samples_conditional.csv --ref data/train_clean.csv
#
# 指标：
#   validity   有效率（RDKit 能解析的比例）
#   uniqueness 有效分子中去重后的比例
#   novelty    不在参考集（默认 test.csv，即模型没见过的分子）中的比例
#   QED / SA / LogP / MW 的均值和分布图
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog('rdApp.error')


def load_smiles(path, col=None):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if col is None:
        col = 'SMILES' if rows and 'SMILES' in rows[0] else \
            ('canonical' if rows and 'canonical' in rows[0] else list(rows[0].keys())[0])
    return [r[col] for r in rows]


def evaluate(generated, reference_smiles):
    mols = [Chem.MolFromSmiles(s) for s in generated]
    valid = [m for m in mols if m is not None]

    validity = len(valid) / len(generated)
    canon_valid = {Chem.MolToSmiles(m) for m in valid}
    uniqueness = len(canon_valid) / len(valid) if valid else 0.0

    ref_set = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in reference_smiles
               if Chem.MolFromSmiles(s) is not None}
    novel = canon_valid - ref_set
    novelty = len(novel) / len(canon_valid) if canon_valid else 0.0

    props = {
        'QED': [QED.qed(m) for m in valid],
        'SA': [sascorer.calculateScore(m) for m in valid],  # 越低越好合成
        'LogP': [Crippen.MolLogP(m) for m in valid],
        'MW': [Descriptors.MolWt(m) for m in valid],
    }
    return validity, uniqueness, novelty, props, canon_valid


def plot_distributions(props, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, vals) in zip(axes.flat, props.items()):
        ax.hist(vals, bins=50, color='#dd8452')
        ax.set_title(f'{name}  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f'分布图已保存: {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated', required=True, help='生成的 SMILES csv（project3/4 的输出）')
    ap.add_argument('--ref', default='data/test.csv',
                    help='新颖性参考集；默认用 test.csv（训练时模型没见过）')
    ap.add_argument('--ref-col', default=None)
    ap.add_argument('--gen-col', default=None)
    args = ap.parse_args()

    generated = load_smiles(args.generated, args.gen_col)
    # test.csv 是原始大文件，只取一部分做参考集即可
    reference = load_smiles(args.ref, args.ref_col)[:100000]

    validity, uniqueness, novelty, props, canon_valid = evaluate(generated, reference)

    print('\n========== 生成分子评价报告 ==========')
    print(f'生成总数:        {len(generated)}')
    print(f'validity  有效率: {validity:.3f}')
    print(f'uniqueness 唯一性: {uniqueness:.3f}')
    print(f'novelty   新颖性: {novelty:.3f}（相对 {args.ref}）')
    if not canon_valid:
        print('没有有效分子！模型可能训练不足（多训几个 epoch 或加大数据量）')
    for name, vals in props.items():
        line = f'{name:<4} '
        if vals:
            line += (f'mean={np.mean(vals):7.2f}  std={np.std(vals):5.2f}  '
                     f'min={np.min(vals):7.2f}  max={np.max(vals):7.2f}')
        print(line)
    print('======================================')

    if not canon_valid:
        return

    stem = args.generated.rsplit('.', 1)[0]
    plot_distributions(props, f'{stem}_eval.png')

    with open(f'{stem}_eval_report.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['n_generated', len(generated)])
        w.writerow(['validity', f'{validity:.4f}'])
        w.writerow(['uniqueness', f'{uniqueness:.4f}'])
        w.writerow(['novelty', f'{novelty:.4f}'])
        for name, vals in props.items():
            w.writerow([f'{name}_mean', f'{np.mean(vals):.4f}'])
    print(f'指标报告已保存: {stem}_eval_report.csv')

    # 顺手把有效分子单独存一份，方便后续对接/筛选
    with open(f'{stem}_valid.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['SMILES'])
        w.writerows([[s] for s in sorted(canon_valid)])
    print(f'有效分子已保存: {stem}_valid.csv')


if __name__ == '__main__':
    main()
