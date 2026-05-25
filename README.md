# 电信用户流失预测 — 风控建模实战

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-purple.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

基于 [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 数据集的端到端流失预测项目。完整覆盖**数据清洗 → 特征工程 → 四模型对比 → KS/PSI 风控评估 → SHAP 可解释性 → 业务召回策略**全链路。

## 核心成果

| 指标 | 表现 | 说明 |
|------|------|------|
| 四模型 AUC | 均达 **0.84+** | 逻辑回归 0.841 / 随机森林 0.844 / XGBoost 0.839 / MLP 0.838 |
| 最高召回率 | **0.813** (MLP) | 较随机触达效率提升约 **3 倍**（0.813 / 0.265） |
| KS 判别力 | 四模型均 **> 0.52** | 全部达到风控建模"优秀"级（> 0.5） |
| PSI 稳定性 | 均 **< 0.011** | XGBoost 修复数据泄露后 PSI 仅 **0.004** |
| 预期覆盖 | **78.6%** 真实流失用户 | 基于 XGBoost 模型 Recall（使用默认阈值 0.5） |

## 数据集

| 属性 | 说明 |
|------|------|
| 来源 | [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 样本量 | 7,043 条 |
| 特征数 | 20 个原始特征（人口统计、账户信息、服务订阅） |
| 目标变量 | Churn（流失率 26.54%，约 3.7:1 类别不平衡） |

## 项目结构

```
├── telco_churn_fixed1.ipynb                              # 完整分析代码
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                  # 原始数据（需从 Kaggle 下载）
├── figures/                                              # 可视化输出（13 张图）
│   ├── 01_churn_distribution.png                         # 流失率饼图
│   ├── 02_numeric_boxplots.png                           # 数值特征按流失状态分组箱线图
│   ├── 03_churn_rate_by_category.png                     # 合同类型 & 网络服务流失率柱状图
│   ├── 04_churn_by_tenure_group.png                      # tenure 分组流失率
│   ├── 05_roc_curves.png                                 # 四模型 ROC 曲线对比
│   ├── 06_shap_bar.png                                   # SHAP 特征重要性 TOP15
│   ├── 07_shap_beeswarm.png                              # SHAP 蜂群图
│   ├── 08_shap_dependence_top3.png                       # TOP3 SHAP 依赖图
│   ├── 09_ks_curves.png                                  # 四模型 KS 曲线分面对比
│   ├── 10_ks_overlay.png                                 # KS 曲线叠加对比
│   ├── 11_ks_bar.png                                     # KS 值汇总柱状图
│   ├── 12_psi_distribution.png                           # PSI 分数分布对比
│   └── 13_psi_bar.png                                    # PSI 值汇总柱状图
└── README.md
```

## 分析流程

### 1. 数据清洗与特征工程

**数据清洗**：

| 问题 | 处理 |
|------|------|
| `TotalCharges` 为 object 类型（含空格） | `pd.to_numeric(errors='coerce')` 转数值，11 个缺失值用中位数填充 |
| `customerID` 无预测价值 | 直接删除 |
| `Churn` 为 Yes/No 文本 | 映射为 1/0 |

**特征工程**：

| 处理类型 | 涉及特征 | 策略 |
|----------|---------|------|
| 二值编码 | Partner, Dependents, PhoneService, PaperlessBilling, gender | Yes/No → 1/0 |
| 三值→二值 | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines | "No internet/phone service" 与 "No" 合并为 0，由哑变量区分 |
| One-Hot | InternetService, Contract, PaymentMethod | drop_first=True |
| 连续保留 | tenure, MonthlyCharges, TotalCharges | 树模型用原值，逻辑回归/MLP 用 StandardScaler |

**多重共线性处理**：TotalCharges 与 tenure × MonthlyCharges 高度相关（r ≈ 1.0），考虑到预测时所有特征均可获取（不构成数据泄露），且对树模型分裂路径有独立贡献，予以保留。

最终特征矩阵：**23 个特征，7,043 条样本**。

### 2. 四模型对比

**统一配置**：分层抽样 80/20 划分，通过 class_weight / scale_pos_weight / pos_weight 处理样本不平衡。

| 模型 | AUC | Recall | F1 | Precision | Accuracy |
|------|-----|--------|----|-----------|----------|
| 逻辑回归 | 0.8413 | 0.7807 | 0.6128 | 0.5043 | 0.7381 |
| **随机森林** | **0.8444** | 0.7219 | **0.6272** | **0.5544** | **0.7722** |
| XGBoost | 0.8389 | 0.7861 | 0.6216 | 0.5140 | 0.7459 |
| MLP (PyTorch) | 0.8384 | **0.8128** | 0.6166 | 0.4967 | 0.7317 |

**关键结论**：

- 四模型 AUC 均在 **0.84 左右**，差距在 0.006 以内，模型选型风险低
- **MLP Recall 达 0.813**，相比随机触达（流失率 26.5%）效率提升约 **3 倍**——即模型筛选的高风险名单中，每 100 个真实流失用户能命中 81 个，而随机抽查仅能命中 27 个
- 随机森林综合精度最优（AUC/F1 双高），适合对误判率敏感的场景
- XGBoost 泛化稳定，适合用于 SHAP 可解释性分析

**模型细节**：

| 模型 | 关键配置 |
|------|---------|
| 逻辑回归 | L2 正则化，liblinear 求解器，class_weight='balanced' |
| 随机森林 | 200 棵树，max_depth=10，class_weight='balanced' |
| XGBoost | 500 轮迭代，learning_rate=0.05，max_depth=5，early_stopping_rounds=30，实际收敛于 161 轮 |
| MLP | [128→64→32→1] + BatchNorm + Dropout(0.3/0.3/0.2)，BCEWithLogitsLoss + pos_weight，Adam + weight_decay=1e-4 |

### 3. KS & PSI — 风控建模核心指标

自实现 KS 与 PSI 计算函数，从判别力和稳定性双维度评估模型，对标风控建模标准。

#### KS (Kolmogorov-Smirnov) — 模型判别能力

KS = max(|TPR - FPR|)，衡量模型区分流失/非流失用户的能力。

| KS 范围 | 评级 | 本项目 |
|---------|------|--------|
| > 0.5 | 优秀 | **四模型全部达标** |
| 0.3 ~ 0.5 | 良好 | — |
| < 0.2 | 较差 | — |

#### PSI (Population Stability Index) — 模型稳定性

PSI = Σ(Actual% - Expected%) × ln(Actual% / Expected%)，衡量训练/测试集分数分布一致性。

| PSI 范围 | 评级 | 本项目 |
|----------|------|--------|
| < 0.1 | 稳定 | **四模型全部 < 0.011** |
| 0.1 ~ 0.25 | 轻微偏移 | — |
| > 0.25 | 不稳定 | — |

#### 综合评估

| 模型 | KS | KS 评级 | PSI | PSI 评级 | Recall | AUC |
|------|-----|---------|-----|----------|--------|-----|
| 随机森林 | **0.5452** | 优秀 | 0.0101 | 稳定 | 0.7219 | 0.8444 |
| XGBoost | 0.5300 | 优秀 | **0.0040** | 稳定 | 0.7861 | 0.8389 |
| MLP | 0.5290 | 优秀 | 0.0100 | 稳定 | **0.8128** | 0.8384 |
| 逻辑回归 | 0.5219 | 优秀 | 0.0079 | 稳定 | 0.7807 | 0.8413 |

> 四模型 **KS 均 > 0.52**（风控优秀线 0.5），**PSI 均 < 0.011**（稳定线 0.1），具备上线部署的模型质量基础。XGBoost PSI 仅 0.004，稳定性在四模型中最优。

#### 数据泄露修复记录

初版 XGBoost 的 early stopping 验证集误用了测试集数据，导致 AUC 虚高至 0.944。修复后从训练集内部切分 15% 作为验证集（4,788 / 846），测试集全程盲测：

| 阶段 | XGBoost AUC | XGBoost KS | XGBoost PSI |
|------|------------|------------|-------------|
| 修复前（数据泄露） | 0.9443 | 0.7662 | ~0.010 |
| **修复后（正确验证）** | **0.8389** | **0.5300** | **0.0040** |

> 修复后 AUC 回落至与其他模型一致的水平，但 PSI 从 ~0.01 进一步降至 0.004，模型稳定性显著提升——验证了"正确的验证策略 → 更真实的泛化能力"。

### 4. SHAP 可解释性分析

使用 SHAP TreeExplainer 对 XGBoost 进行全局解释，量化每个特征对流失预测的边际贡献。

#### TOP 3 关键驱动因子

| 排名 | 特征 | SHAP 重要性 | 业务解读 |
|------|------|------------|---------|
| 1 | **合同类型** (Contract_Two year) | 0.717 | 两年合同是最强保留信号——签订长期合约的用户流失概率大幅降低 |
| 2 | **在网时长** (tenure) | 0.452 | 在网越久、黏性越强——新用户前 12 个月是流失高发期 |
| 3 | **光纤服务** (InternetService_Fiber optic) | 0.276 | 光纤用户月费更高、流失风险显著高于 DSL 用户 |

#### 完整 TOP 10

| 排名 | 特征 | SHAP 重要性 | 影响方向 |
|------|------|------------|---------|
| 1 | Contract_Two year | 0.7174 | 两年合同大幅降低流失 |
| 2 | tenure | 0.4518 | 在网时长越长风险越低 |
| 3 | Contract_One year | 0.3427 | 一年合同也显著优于按月 |
| 4 | InternetService_Fiber optic | 0.2757 | 光纤用户流失风险更高 |
| 5 | MonthlyCharges | 0.2688 | 月费越高流失倾向越强 |
| 6 | TotalCharges | 0.2436 | 历史消费额低 → 风险高 |
| 7 | PaymentMethod_Electronic check | 0.1558 | 电子支票用户风险偏高 |
| 8 | InternetService_No | 0.1129 | 无互联网服务用户特征 |
| 9 | PaperlessBilling | 0.1122 | 电子账单用户风险略高 |
| 10 | StreamingMovies | 0.1058 | 流媒体订阅有边际保护作用 |

### 5. 业务策略建议

#### 高风险用户画像（需重点干预）

- 按月合同 + 在网时长 < 12 个月 + 月费 > 70 元
- 光纤用户 + 未开通安全/技术支持等增值服务
- 使用电子支票支付

#### 「长期合约激励 + 重点用户定向召回」运营策略

基于 XGBoost 模型（Recall 0.786），使用默认阈值 0.5 可覆盖 **78.6% 的真实流失用户**——即每 100 个最终流失的用户中，模型能提前识别 79 人，为定向干预争取窗口期。

1. **推动月合同 → 长期合同转化**（SHAP 影响最大）
   - 年付折扣（8-9 折）、忠诚度积分、专属优惠券
2. **新用户前 12 个月重点关怀**
   - 入网引导、第 30/90 天满意度回访、专属客服通道
3. **高月费用户定向优惠**
   - 套餐捆绑折扣、阶梯返利、降档宽限期
4. **光纤用户专项留存**
   - 网络质量调研、免费技术支持升级、增值服务试用
5. **引导电子支票 → 自动扣款**
   - 自动扣款立减 5 元/月、免手续费激励

## 技术栈

| 层级 | 工具 | 用途 |
|------|------|------|
| 数据处理 | Pandas, NumPy | 数据清洗、特征工程、多重共线性分析 |
| 可视化 | Matplotlib, Seaborn | EDA、13 张模型评估/风控图表 |
| 传统模型 | Scikit-learn | 逻辑回归、随机森林 |
| 梯度提升 | XGBoost | early stopping + scale_pos_weight |
| 深度学习 | PyTorch | 三层 MLP + BatchNorm + Dropout |
| 可解释性 | SHAP | TreeExplainer、summary_plot、dependence_plot |
| 风控指标 | 自实现 | KS 统计量、PSI 群体稳定性指数 |

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction

# 2. 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch shap

# 3. 下载数据集并放入项目根目录
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. 运行 Notebook
jupyter notebook telco_churn_fixed1.ipynb
```

## 关键设计决策与踩坑记录

- **Early Stopping 数据泄露修复**：验证集必须从训练集内部切分，不能使用测试集。修复后 AUC 从虚高 0.944 回落至 0.839，PSI 从 ~0.01 降至 0.004，模型的泛化评估回归真实水平
- **多重共线性处理**：TotalCharges ≈ tenure × MonthlyCharges（r ≈ 1.0），考虑到预测时所有特征均可获取（不构成数据泄露），且对树模型的分裂路径有独立贡献，予以保留
- **tenure_group 不进模型**：分组哑变量与连续 tenure 完全冗余，仅在 EDA 阶段用于可视化
- **BCEWithLogitsLoss > BCELoss**：集成 Sigmoid 的损失函数在数值稳定性上优于分步计算，避免 logit 极值溢出
- **四类模型覆盖**：线性 / 树 / 梯度提升 / 神经网络，为不同场景的模型选型提供充分参照

## License

MIT
