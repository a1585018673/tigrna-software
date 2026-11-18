# 来源与建模口径

## 论文

Faure et al., “TIGR-Tas: A family of modular RNA-guided DNA-targeting systems in prokaryotes and their viruses,” Science 388, eadv9789 (2025), DOI: 10.1126/science.adv9789。

与建模直接相关的论文信息：

- Tas 蛋白结合约 36 nt 的 tigRNA；TaTIGR/FpTIGR 的可变 split spacers 合计 18 nt，并通过两段 spacer 协同识别 DNA 双链。
- 论文的随机侧翼筛选没有观察到强 5′或 3′ TAM/PAM 富集，因此两侧各 1 nt 只作为弱序列特征，不设置硬性 motif 规则。
- 靠近切割位点的两个 seed 位置对 mismatch 更敏感；模型用位置碱基和位置二核苷酸特征保留这种可能性，但当前数据不足以可靠解析单位置因果效应。
- 论文的人细胞实验使用 HEK293FT；当前工作簿包含 293T 和 N2a，且使用优化体 336-172，因此当前模型是本实验体系的经验校准，不能视为论文原始 TaTasR/ParTasR 的通用模型。

## 工作簿

- 输入设计：1 nt `5'TAM` + 18 nt `target` + 1 nt `3'TAM`。
- 主标签：`indel%(336-172)` 三个位置中所有非空测量的算术均值。
- 空白单元格表示缺失，不按 0 处理；只有显式数值 0 才作为零效率。
- 293T 有 98 条优化体标签、35 个独立位点；N2a 有 27 条优化体标签，但全部来自一个 PCSK 位点，因此不能验证 N2a 跨位点泛化。

## 蛋白序列

本实验使用的氨基酸序列保存在 `protein_sequence.fasta`。由于所有样本使用同一蛋白，该序列是模型版本元数据，不是可学习的样本特征。
