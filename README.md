# 电信用户流失预测项目

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

基于 [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 数据集的端到端机器学习项目。对比三种模型（逻辑回归、随机森林、XGBoost）的性能，并使用 SHAP 值解释流失的关键驱动因素，输出可落地的运营召回建议。

## 项目结构

```
电信用户流失预测项目/
├── telco_churn_fixed.ipynb                         # 完整分析代码
├── WA_Fn-UseC_-Telco-Customer-Churn.csv            # 原始数据（需从 Kaggle 下载）
├── figures/                                        # 可视化输出
│   ├── 01_churn_distribution.png                   # 流失率饼图
│   ├── 02_numeric_boxplots.png                     # 数值特征按流失状态分组箱线图
│   ├── 03_churn_rate_by_category.png               # 合同类型 & 网络服务流失率柱状图
│   ├── 04_churn_by_tenure_group.png                # tenure 分组流失率柱状图
│   ├── 05_roc_curves.png                           # 三模型 ROC 曲线对比
│   ├── 06_shap_bar.png                             # SHAP 特征重要性 TOP15
│   ├── 07_shap_beeswarm.png                        # SHAP 蜂群图
│   └── 08_shap_dependence_top3.png                 # TOP3 特征 SHAP 依赖图
└── README.md
```

## 数据集概览

- **样本量**: 7,043 条用户记录
- **特征数**: 20 个原始特征（人口统计、账户信息、服务订阅）
- **目标变量**: Churn（Yes/No）
- **整体流失率**: 26.54%（约 3.7:1 的类别不平衡）

## 分析流程

### 1. 数据清洗

| 问题 | 处理方式 |
|------|---------|
| `TotalCharges` 为字符串类型（含空格） | `pd.to_numeric(errors='coerce')` 转数值，11 个缺失值用中位数填充 |
| `customerID` 是主键，无预测价值 | 直接删除 |
| `Churn` 为 Yes/No 文本 | 映射为 1/0 |

### 2. 探索性数据分析（EDA）

- **流失率饼图** — 直观展示 26.5% vs 73.5% 的类别分布
- **数值特征箱线图** — tenure、MonthlyCharges、TotalCharges 按流失/留存分组对比。流失用户典型特征：在网月数短、月费偏高
- **分类特征柱状图** — 按月合同流失率（约 42%）远高于一年合同（约 11%）和两年合同（约 3%）；光纤用户流失率显著高于 DSL 用户
- **tenure 分组分析** — 0-12 月新用户流失率约 40%，48+ 月老用户仅约 10%（仅供 EDA 参考，不进模型）
- **相关性热力图** — TotalCharges 与 tenure × MonthlyCharges 近似完全相关（r ≈ 1.0）

### 3. 特征工程

| 处理方式 | 涉及特征 |
|---------|---------|
| Yes/No → 1/0 | Partner, Dependents, PhoneService, PaperlessBilling, gender |
| 三值合并为二值（含 "No internet/phone service" 统一映射为 0） | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines |
| One-Hot 编码（drop_first=True） | InternetService, Contract, PaymentMethod |

**最终特征矩阵**: 23 个特征，7,043 条样本。

> `tenure_group` 仅在 EDA 阶段用于可视化，不进入特征矩阵，避免与连续 tenure 冗余。

### 4. 三模型对比

**统一配置**: 分层抽样（stratify=y）80/20 划分；逻辑回归用 `StandardScaler` + `class_weight='balanced'`；树模型用 `class_weight='balanced'` / `scale_pos_weight` 处理不平衡。

| 模型 | AUC | F1 | Recall | Precision | Accuracy |
|------|-----|----|--------|-----------|----------|
| 逻辑回归 | 0.8413 | 0.6128 | 0.7807 | 0.5043 | 0.7381 |
| 随机森林 | **0.8444** | **0.6272** | 0.7219 | **0.5544** | **0.7722** |
| XGBoost | 0.8389 | 0.6216 | **0.7861** | 0.5140 | 0.7459 |

**模型选择**: 三模型 AUC 最大差距仅 0.0055，在统计噪声范围内（标准误差约 ±0.011），本质上是并列水平。考虑到流失预测场景下 **Recall 是最关键的指标**（漏掉一个真实流失用户的代价远大于误判一个留存用户），最终选用 **XGBoost**（Recall 0.7861）进行后续 SHAP 分析。

### 5. SHAP 可解释性分析（TOP10 特征）

| 排名 | 特征 | SHAP 重要性 | 解读 |
|------|------|------------|------|
| 1 | Contract_Two year | 0.7174 | 两年合同是最强保留信号，大幅降低流失概率 |
| 2 | tenure | 0.4518 | 在网时长越长，流失风险越低 |
| 3 | Contract_One year | 0.3427 | 一年合同也比按月合同显著稳定 |
| 4 | InternetService_Fiber optic | 0.2757 | 光纤用户流失风险高于 DSL 用户 |
| 5 | MonthlyCharges | 0.2688 | 月费越高，流失倾向越强 |
| 6 | TotalCharges | 0.2436 | 与 tenure 协同影响，历史消费额越低风险越高 |
| 7 | PaymentMethod_Electronic check | 0.1558 | 电子支票支付用户流失风险偏高 |
| 8 | InternetService_No | 0.1129 | 无互联网服务用户特征 |
| 9 | PaperlessBilling | 0.1122 | 电子账单用户流失风险略高 |
| 10 | StreamingMovies | 0.1058 | 流媒体服务使用情况有边际影响 |

### 6. 业务结论与召回建议

#### 高风险用户画像（需重点干预）

- 按月合同 + 在网时长 < 12 个月 + 月费 > 70 元
- 光纤用户 + 未开通安全/技术支持等增值服务
- 使用电子支票支付

#### 召回建议（按优先级排序）

1. **推动月合同 → 长期合同转化**：年付折扣（8-9 折）、忠诚度积分、专属优惠券
2. **新用户前 12 个月重点关怀**：入网引导、第 30/90 天满意度回访、专属客服通道
3. **高月费用户定向优惠**：套餐捆绑折扣、阶梯返利、降档宽限期
4. **光纤用户专项留存**：网络质量调研、免费技术支持升级、增值服务试用
5. **引导电子支票 → 自动扣款**：自动扣款立减 5 元/月、免手续费激励

## 环境配置

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

运行 Notebook：

```bash
jupyter notebook telco_churn_fixed.ipynb
```

## 关键设计决策

- **保留 TotalCharges**: 虽然是 tenure × MonthlyCharges 的派生变量，但所有特征在预测时均已知，不构成数据泄露；且对树模型的分裂路径有独立贡献
- **tenure_group 不进模型**: 分组哑变量与连续 tenure 完全冗余，仅在 EDA 阶段用于可视化解读
- **XGBoost early stopping**: 从训练集内部切 15% 作为验证集，测试集全程保持盲测，零数据泄露
- **三模型对比而非单模型**: 提供基准参照，避免单一模型的偶然性，也让模型选择有据可依

## 开源协议

MIT
