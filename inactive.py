import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv("无规律.csv", encoding="gbk")
    df.dropna(inplace=True)
    df.columns = [
        'Achilles tendon stress',
        'Ankle plantar/dorsiflexion angle',
        'Ankle in/eversion angle',
        'Ankle in/external rotation angle',
        'Ankle plantar/dorsiflexion moment',
        'Ankle in/eversion moment',
        'Ankle power',
        'A/P GRF',
        'M/L GRF',
        'Hip flex/extension angle',
        'Hip in/external rotation angle',
        'Hip in/external rotation moment',
        'Knee flex/extension angle',
        'Knee in/external rotation angle',
        'Ipsi/contralateral pelvis rotation',
        'EMG activation for peroneus longus'
    ]
    return df

data = load_data()

# 特征和标签
feature_columns = data.columns[1:]
X = data[feature_columns]
y = data["Achilles tendon stress"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=1,
    min_child_weight=4,
    learning_rate=0.03,
    n_estimators=500,
    subsample=0.8,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Streamlit 前端
st.title("Achilles Tendon Stress Prediction")
st.write("""
This application predicts Achilles tendon stress during the start running phase.  
After entering your motion parameters below, the model will estimate your tendon stress.  
If the predicted value is high, you may consider adjusting your running posture to reduce stress and prevent injuries.
""")

# 侧边栏输入
st.sidebar.header("Input Parameters")
input_values = []
for col in feature_columns:
    # 针对不同变量设置合理范围
    if "moment" in col.lower():
        val = st.sidebar.slider(col, -20.0, 20.0, 0.5)
    elif "angle" in col.lower() or "rotation" in col.lower() or "pelvis" in col.lower():
        val = st.sidebar.slider(col, -100.0, 100.0, 0.0)
    elif "power" in col.lower() or "grf" in col:
        val = st.sidebar.slider(col, -200.0, 200.0, 50.0)
    elif "emg" in col.lower():
        val = st.sidebar.slider(col, 0.0, 1.0, 0.5)
    else:
        val = st.sidebar.slider(col, -100.0, 100.0, 0.0)
    input_values.append(val)

user_input_df = pd.DataFrame([input_values], columns=feature_columns)

# 预测
predicted_stress = model.predict(user_input_df)

# 显示结果
st.subheader("Prediction Result")
st.write(f"**Predicted Achilles Tendon Stress:** `{predicted_stress[0]:.2f}`")
