import pandas as pd
import numpy as np
from pyecharts.charts import Bar, Page, Pie, WordCloud
from pyecharts import options as opts
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data=pd.read_csv('./data/job.csv')
print(data.head())
print(data.shape)

#检查缺失值
missing_values = data.isnull().sum()
data = data.dropna(axis=0)
#检查重复值
duplicate_rows = data.duplicated().sum()
data = data.drop_duplicates()
print(data.shape)


def convert_salary_range(salary_str):
    # 使用正则表达式提取数字和单位
    pattern = re.compile(r'([\d.]+)(千|万)?-([\d.]+)(千|万)?')
    match = pattern.match(salary_str)

    # 如果匹配失败，返回 NaN 值
    if match is None:
        return pd.Series([np.nan, np.nan], index=['最低薪资', '最高薪资'])

    # 提取匹配的组
    lower, lower_unit, upper, upper_unit = match.groups()

    # 转换为数值
    lower = float(lower) * 1000 if lower_unit == '千' else float(lower) * 10000
    upper = float(upper) * 1000 if upper_unit == '千' else float(upper) * 10000

    return pd.Series([lower, upper], index=['最低薪资', '最高薪资'])

# 使用函数转换 "薪资范围" 列，并将计算出的最低薪资和最高薪资添加到数据集中
data_salary_bounds = data["薪资范围"].apply(convert_salary_range)
data = pd.concat([data, data_salary_bounds], axis=1)

print(data.head())

data['平均薪资'] = (data['最低薪资'] + data['最高薪资']) / 2

# 计算每个城市的平均薪资
average_salary_by_city = data.groupby('city')['平均薪资'].mean().sort_values(ascending=False)

import os

# 创建 output 目录
os.makedirs('output', exist_ok=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=average_salary_by_city.values, y=average_salary_by_city.index, hue=average_salary_by_city.index, palette="viridis", legend=False)
plt.xlabel('平均薪资')
plt.ylabel('城市')
plt.title('按城市的平均薪资分布')
plt.tight_layout()
plt.savefig('output/按城市平均薪资分布.png', dpi=150)
plt.close()

# 计算每个城市的薪资分布
salary_distribution_by_city = data.groupby('city')['平均薪资'].describe()
# 使用 seaborn 创建箱线图
plt.figure(figsize=(10, 6))  # 设置图表大小
sns.boxplot(x='city', y='平均薪资', data=data)
plt.yscale('log')  # 使用对数尺度
plt.xticks(rotation=45)  # 旋转 x 轴标签，避免重叠
plt.title('薪资分布箱线图（对数尺度）')
plt.show()

average_salary_by_industry = data.groupby('公司类型')['平均薪资'].mean().sort_values(ascending=False)

# 使用 seaborn 创建条形图
plt.figure(figsize=(10, 6))  # 设置图表大小
sns.barplot(x=average_salary_by_industry.values, y=average_salary_by_industry.index)
plt.xlabel('平均薪资')
plt.title('不同公司类型的平均薪资')
plt.show()

average_salary_by_education = data.groupby('学历要求')['平均薪资'].mean().sort_values(ascending=False)
# 使用 seaborn 创建条形图
plt.figure(figsize=(10, 6))  # 设置图表大小
sns.barplot(x=average_salary_by_education.values, y=average_salary_by_education.index)
plt.xlabel('平均薪资')
plt.title('不同学历要求的平均薪资')
plt.show()