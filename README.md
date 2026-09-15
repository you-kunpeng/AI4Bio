# AI4Bio：AI 分子生成学习路线（6 个阶段项目）

按综述骨架"分子表示 → 数据资源 → 生成模型 → 条件生成 → 评价体系 → 复现"组织，
每个阶段一个可独立运行的 Python 脚本，产出落到当前目录 / data 目录。

依赖：`pip install rdkit pandas requests matplotlib tqdm torch`（本机已装，torch 为 CPU 版）。

## ⚠️ 数据下载的坑（为什么之前"清洗后数量: 0"）

MOSES 仓库（molecularsets/moses）里的 `train.csv` 等文件是 **Git LFS 文件**。
`raw.githubusercontent.com` 和 jsdelivr 返回的只是 ~134 字节的**指针文本**，
第一行是 `version https://git-lfs.github.com/spec/v1`——旧脚本把它当成 CSV 解析，
于是出现"原始列名 = 指针第一行、2 行数据、清洗后 0 条、词表只有 4 个特殊符号"，
后续 RDKit 解析 `oid`、`size`（指针文件第二三行的开头）才报 SMILES 语法错误。

**正确做法**：用 LFS 实体地址 `https://media.githubusercontent.com/media/molecularsets/moses/master/data/<文件名>`
（project2 已改用该源并自动校验文件头，~68MB 才是真实数据）。

## 阶段一：RDKit 基础

`project1_rdkit_basics.py`：读 SMILES、画分子、算 QED / LogP / SA，保存网格图和性质表。
```bash
python project1_rdkit_basics.py
```

## 阶段二：数据资源与清洗

`project2_data_prepare.py`：下载 MOSES（train/test/test_scaffolds）→ RDKit 清洗
（canonical 化、去无效、去重）→ 性质分布图（MW/QED/LogP/SA）→ 构建 `vocab.pkl`。
```bash
python project2_data_prepare.py --sample 50000   # 快速版：每个 split 清洗前 5 万条
python project2_data_prepare.py                  # 全量：train 约 193 万条，清洗约 10 分钟
```
产出：`data/*_clean.csv`、`vocab.pkl`、`project2_property_distributions.png`。

## 阶段三：无条件生成（SMILES LSTM）

`project3_smiles_lstm.py`：字符级 2 层 LSTM，逐字符生成分子。
```bash
python project3_smiles_lstm.py --mode train --epochs 5          # 全量训练约 20~40 分钟(CPU)
python project3_smiles_lstm.py --mode train --epochs 2 --max-molecules 20000  # 快速试跑
python project3_smiles_lstm.py --mode sample --n 1000           # 采样 1000 个分子
```
产出：`project3_lstm.pt`、`data/samples_lstm.csv`。

## 阶段四：条件生成（QED 条件 LSTM）

`project4_conditional_gen.py`：把 QED 分成 5 档做成条件 token 拼在序列开头，
可以指定"给我生成高类药性（QED 第 4 档）的分子"。
```bash
python project4_conditional_gen.py --mode train --epochs 5
python project4_conditional_gen.py --mode sample --n 200 --qed-bin 4
```
产出：`project4_cond_lstm.pt`、`data/samples_conditional.csv`。

**对接筛选（可选）**：`pip install vina meeko`，用 `mk_prepare_receptor.py` 把 PDB 受体转 pdbqt，
配体用 `Meeko` 从 SMILES 转 pdbqt 后批量对接，按打分取 Top 100。
路线图里的靶点条件生成（DiffSBDD/TargetDiff）需要 3D 口袋数据 + GPU，建议作为后续课题。

## 阶段五：评价体系

`project5_evaluation.py`：输入任何一批生成的 SMILES，输出
validity / uniqueness / novelty / QED / SA / LogP / MW + 分布图 + 报告 csv。
```bash
python project5_evaluation.py --generated data/samples_lstm.csv
python project5_evaluation.py --generated data/samples_conditional.csv
```
新颖性默认相对 `data/test.csv` 计算（模型训练时没见过的分子）。

## 阶段六：复现（SMILES VAE）

`project6_smiles_vae.py`：复现 ChemVAE 的最小版本——LSTM 编码到 64 维潜在空间、
重参数化、KL 散度、LSTM 解码、LogP 回归头；训练完可做**潜在空间分子插值**。
```bash
python project6_smiles_vae.py --mode train --epochs 5
python project6_smiles_vae.py --mode interpolate --n 5   # 两个真实分子之间插值
```
产出：`project6_vae.pt`、`data/vae_interpolation.csv`。

## 推荐阅读顺序

Segler LSTM → Junction Tree VAE → MolGAN → GraphVAE → Chemformer → REINVENT4
→ GeoDiff / EDM → DiffSBDD / TargetDiff → MolGPT / DrugGPT。

## 已验证

- project2（--sample 50000）：下载 + 清洗 + 词表 ✓（5 万条全部有效，词表 29 字符）
- project3/4：训练 ✓、批量采样 ✓（采样已改为一次并行 64 条，CPU 上 1000 个分子只需几秒）
- project5：评估 ✓（0 有效分子时会给出提示而不是崩溃）
- project6：训练 ✓、插值 ✓

提示：本机还有其他训练任务在跑时会抢占 CPU，训练请错峰；GPU 版（`pip install torch --index-url https://download.pytorch.org/whl/cu121`）会快一个数量级。
