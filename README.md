### 职位数据分析

基于招聘网站的职位数据（job.csv），进行薪资趋势与影响因素分析。

#### 数据准备

```bash
mkdir data
# 将 job.csv 放入 data/ 目录
```

#### 环境要求

```bash
conda create -n job_analysis python=3.11.15
conda activate job_analysis
pip install pandas numpy seaborn matplotlib scikit-learn pyecharts
```

#### 运行

```bash
python main.py
```

图表输出至 `output/` 目录。

#### 分析内容

| 维度 | 输出文件 | 说明 |
|------|----------|------|
| 城市 | 按城市平均薪资分布.png | 各城市平均薪资对比 |
| 城市 | 各城市薪资分布箱线图.png | 薪资分布与异常值 |
| 公司类型 | 不同公司类型平均薪资.png | 民营/国企/外企等对比 |
| 学历要求 | 不同学历要求平均薪资.png | 学历对薪资的影响 |

#### TODO

- [ ] 爬取招聘数据并放入 `data/` 目录
- [ ] 使用模型拟合薪资数据
- [ ] 输入条件预测薪资
