# project1_rdkit_basics.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw, rdMolDescriptors

# SA_Score 在 RDKit 的 Contrib 里，不同安装方式可能没有
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    def calc_sa(mol):
        return sascorer.calculateScore(mol)
except Exception as e:
    print('SA_Score 不可用，将返回 NaN:', e)
    def calc_sa(mol):
        return float('nan')

smiles_list = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',          # 阿司匹林
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',      # 布洛芬
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',       # 咖啡因
    'C1=CC=C(C=C1)C2=CC=CC=C2',           # 联苯
    'CCN(CC)CC',                           # 三乙胺
    'C1CCCCC1',                            # 环己烷
    'c1ccccc1',                            # 苯
    'CC(=O)NC1=CC=C(C=C1)O',              # 对乙酰氨基酚
]

rows = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print('无效 SMILES:', smi)
        continue
    rows.append({
        'SMILES': smi,
        'Canonical_SMILES': Chem.MolToSmiles(mol, canonical=True),
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'QED': QED.qed(mol),
        'SA': calc_sa(mol),
    })

df = pd.DataFrame(rows)
print(df)
df.to_csv('project1_molecule_properties.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].hist(df['MW'], bins=10, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Molecular Weight')
axes[0, 1].hist(df['LogP'], bins=10, color='salmon', edgecolor='black')
axes[0, 1].set_title('LogP')
axes[1, 0].hist(df['QED'], bins=10, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('QED')
axes[1, 1].hist(df['SA'], bins=10, color='gold', edgecolor='black')
axes[1, 1].set_title('SA Score')
plt.tight_layout()
plt.savefig('project1_property_distributions.png', dpi=200)
plt.close()

mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
img = Draw.MolsToGridImage(mols[:8], molsPerRow=4, subImgSize=(250, 250))
img.save('project1_molecule_grid.png')

print('完成：project1_molecule_properties.csv, project1_property_distributions.png, project1_molecule_grid.png')