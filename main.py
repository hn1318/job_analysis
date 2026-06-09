import pandas as pd
import numpy as np
from pyecharts.charts import Page, Pie, WordCloud
from pyecharts import options as opts
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('Agg')

data = pd.read_csv('./data/job.csv')
print(data.head())
print(data.shape)

# 检查缺失值
missing_values = data.isnull().sum()
data = data.dropna(axis=0)
# 检查重复值
duplicate_rows = data.duplicated().sum()
data = data.drop_duplicates()
print(data.shape)


def convert_salary_range(salary_str):
    pattern = re.compile(r'([\d.]+)([千万])?-([\d.]+)([千万])?')
    match = pattern.match(salary_str)

    if match is None:
        return pd.Series([np.nan, np.nan], index=['最低薪资', '最高薪资'])

    # 提取匹配的组
    lower, lower_unit, upper, upper_unit = match.groups()

    # 转换为数值（单位：元）
    # 有"千"乘1000，有"万"乘10000，无单位则已经是元
    lower_val = float(lower)
    upper_val = float(upper)
    if lower_unit == '千':
        lower = lower_val * 1000
    elif lower_unit == '万':
        lower = lower_val * 10000
    else:
        lower = lower_val  # 已经是元，不要乘

    if upper_unit == '千':
        upper = upper_val * 1000
    elif upper_unit == '万':
        upper = upper_val * 10000
    else:
        upper = upper_val  # 已经是元，不要乘

    return pd.Series([lower, upper], index=['最低薪资', '最高薪资'])


data_salary_bounds = data["薪资范围"].apply(convert_salary_range)
data = pd.concat([data, data_salary_bounds], axis=1)

print(data.head())

data['平均薪资'] = (data['最低薪资'] + data['最高薪资']) / 2


average_salary_by_city = data.groupby('city')['平均薪资'].mean().sort_values(ascending=False)

import os

# 创建 output 目录
os.makedirs('output', exist_ok=True)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=average_salary_by_city.values, y=average_salary_by_city.index,
                 hue=average_salary_by_city.index, palette="viridis", legend=False)
# 添加数值标签
for i, v in enumerate(average_salary_by_city.values):
    ax.text(v + 50, i, f'{v:,.0f}', va='center', fontsize=9)
plt.xlabel('平均薪资')
plt.ylabel('城市')
plt.title('按城市的平均薪资分布', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('output/按城市平均薪资分布.png', dpi=150)
plt.close()

city_order = data.groupby('city')['平均薪资'].median().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='city', y='平均薪资', data=data, order=city_order, hue='city', palette="viridis", legend=False)
plt.yscale('log')
plt.xticks(rotation=45)
plt.xlabel('城市', fontsize=12)
plt.ylabel('平均薪资', fontsize=12)
plt.title('各城市薪资分布箱线图（对数尺度）', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output/各城市薪资分布箱线图.png', dpi=150)
plt.close()

average_salary_by_industry = data.groupby('公司类型')['平均薪资'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=average_salary_by_industry.values, y=average_salary_by_industry.index,
                 hue=average_salary_by_industry.index, palette="viridis", legend=False)
for i, v in enumerate(average_salary_by_industry.values):
    ax.text(v + 50, i, f'{v:,.0f}', va='center', fontsize=9)
plt.xlabel('平均薪资', fontsize=12)
plt.ylabel('公司类型', fontsize=12)
plt.title('不同公司类型的平均薪资', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('output/不同公司类型平均薪资.png', dpi=150)
plt.close()


average_salary_by_education = data.groupby('学历要求')['平均薪资'].mean().sort_values(ascending=False)

# 注意：必须按排序后的顺序指定 y 轴类别顺序，否则 seaborn 会按字母排序
education_order = average_salary_by_education.index.tolist()
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=average_salary_by_education.values, y=average_salary_by_education.index,
                 hue=average_salary_by_education.index, order=education_order, palette="viridis", legend=False)
for i, v in enumerate(average_salary_by_education.values):
    ax.text(v + 50, i, f'{v:,.0f}', va='center', fontsize=9)
plt.xlabel('平均薪资', fontsize=12)
plt.ylabel('学历要求', fontsize=12)
plt.title('不同学历要求的平均薪资', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('output/学历与薪资关系分析.png', dpi=150)
plt.close()


average_salary_by_experience = data.groupby('工作经验')['平均薪资'].mean().sort_values(ascending=False)
# 使用 seaborn 创建条形图
plt.figure(figsize=(10, 6))  # 设置图表大小
sns.barplot(x=average_salary_by_experience.values, y=average_salary_by_experience.index)
plt.xlabel('平均薪资')
plt.title('不同工作经验要求的平均薪资')
plt.tight_layout()
plt.savefig('output/不同工作经验要求平均薪资.png', dpi=150)
plt.close()

# 填充 NaN + 对全量类别特征做独热编码
data['最低薪资'] = data['最低薪资'].fillna(data['最低薪资'].median())
data['最高薪资'] = data['最高薪资'].fillna(data['最高薪资'].median())
data['平均薪资'] = (data['最低薪资'] + data['最高薪资']) / 2
data['平均薪资'] = data['平均薪资'].fillna(data['平均薪资'].mean())

categorical_cols = ['city', '工作经验', '学历要求', '公司类型', '公司规模', '职位名称', '公司名称']
data_encoded = pd.get_dummies(data, columns=categorical_cols)
features_to_exclude = ['平均薪资', '薪资范围', '岗位标签', '最低薪资', '最高薪资', '地点'] + categorical_cols
X = data_encoded.drop(columns=features_to_exclude, errors='ignore')
X = X.fillna(X.mean())
y = data['平均薪资']

# 保存特征列名（用于后续预测时对齐）
feature_columns = X.columns.tolist()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"X contains NaN: {X.isna().any().any()}")
print(f"y contains NaN: {y.isna().any()}")

# ============================================================
# 多模型对比：引入多种回归算法，选出最佳模型
# ============================================================

# Ridge 超参调优
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_best = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_best.fit(X_train, y_train)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': ridge_best.best_estimator_,
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF Kernel)': SVR(kernel='rbf'),
}

results = []
trained_models = {}
scalers = {}

print("\n" + "=" * 70)
print("多模型训练与对比...")
print("=" * 70)

for name, model in models.items():
    start = time.time()

    if name.startswith('SVR'):
        # SVR 对特征尺度敏感，需要标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        scalers[name] = scaler
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scalers[name] = None

    elapsed = time.time() - start
    trained_models[name] = model
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({'模型': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2, '耗时(秒)': round(elapsed, 2)})
    print(f" >{name:30s}  RMSE={rmse:>10.2f}  MAE={mae:>10.2f}  R²={r2:.4f}  ({elapsed:.1f}s)")

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)

print("\n" + "=" * 70)
print("各模型表现对比（按 RMSE 升序排列）")
print("=" * 70)
print(results_df.to_string(index=False))

# 取 RMSE 最低的模型作为最佳模型
best_model_name = results_df.loc[0, '模型']
best_model = trained_models[best_model_name]
best_scaler = scalers[best_model_name]

print(f"\n{'=' * 70}")
print(f"最佳模型: {best_model_name}")
print(f"   RMSE: {results_df.loc[0, 'RMSE']:,.2f}")
print(f"   MAE:  {results_df.loc[0, 'MAE']:,.2f}")
print(f"   R²:   {results_df.loc[0, 'R²']:.4f}")
print(f"{'=' * 70}")


# ============================================================
# 真正的薪资预测函数（使用最佳模型）
# ============================================================
def predict_salary(education, experience, desired_city, desired_company_type,
                   desired_position, company_scale=None, company_name=None):
    """
    使用训练好的最佳模型预测平均薪资

    参数:
        education: 学历要求（如 本科）
        experience: 工作经验（如 1-3年）
        desired_city: 城市（如 上海）
        desired_company_type: 公司类型（如 外商独资）
        desired_position: 职位名称（如 数据分析师）
        company_scale: 公司规模（如 500-999人，可选）
        company_name: 公司名称（可选）
    """
    user_data = pd.DataFrame([{
        '学历要求': education,
        '工作经验': experience,
        'city': desired_city,
        '公司类型': desired_company_type,
        '职位名称': desired_position,
        '公司规模': company_scale if company_scale else '',
        '公司名称': company_name if company_name else '',
    }])

    # 与训练时相同的独热编码
    user_encoded = pd.get_dummies(user_data, columns=categorical_cols)
    user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)

    # 如果是 SVR 需要用相同的 scaler 做缩放
    if best_scaler is not None:
        user_encoded_scaled = best_scaler.transform(user_encoded)
        prediction = best_model.predict(user_encoded_scaled)[0]
    else:
        prediction = best_model.predict(user_encoded)[0]

    return round(prediction)


# 交互式薪资预测
print(f"\n{'=' * 50}")
print("薪资预测工具（基于最佳模型）")
print("=" * 50)
education = input("请输入您的学历（例如：本科）: ")
experience = input("请输入您的工作经验（例如：1-3年）: ")
desired_city = input("请输入您希望工作的城市（例如：上海）: ")
desired_company_type = input("请输入您希望的公司类型（例如：国企）: ")
desired_position = input("请输入您希望的职位名称（例如：数据分析师）: ")

predicted_salary = predict_salary(
    education, experience, desired_city, desired_company_type, desired_position
)

print(f"\n{'=' * 50}")
print(f"基于您提供的信息，预计平均薪资为: {predicted_salary:,.0f} 元/月")
print(f"   （使用模型: {best_model_name}）")
print(f"{'=' * 50}")
