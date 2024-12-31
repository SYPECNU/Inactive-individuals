import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap

# 载入数据
@st.cache_data
def load_data():
    data1 = pd.read_csv("无规律.csv", encoding='gbk')
    data1.dropna(inplace=True)
    data1.columns = ['Achilles tendon stress', 'Ankle plantar/dorsiflexion angle', 'Ankle in/eversion angle', 
                     'Ankle in/external rotation angle', 'Ankle plantar/dorsiflexion moment', 'Ankle in/eversion moment', 
                     'Ankle power', 'A/P GRF', 'M/L GRF', 'Hip flex/extension angle', 'Hip in/external rotation angle', 
                     'Hip in/external rotation moment', 'Knee flex/extension angle', 'Knee in/external rotation angle', 
                     'Ipsi/contralateral pelvis rotation', 'EMG activation for peroneus longus']
    return data1

data1 = load_data()

# 提取特征和标签
X = data1[['Ankle plantar/dorsiflexion angle', 'Ankle in/eversion angle', 'Ankle in/external rotation angle', 
           'Ankle plantar/dorsiflexion moment', 'Ankle in/eversion moment', 'Ankle power', 'A/P GRF', 'M/L GRF', 
           'Hip flex/extension angle', 'Hip in/external rotation angle', 'Hip in/external rotation moment', 
           'Knee flex/extension angle', 'Knee in/external rotation angle', 'Ipsi/contralateral pelvis rotation', 
           'EMG activation for peroneus longus']]

y = data1[['Achilles tendon stress']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, min_child_weight=4, 
                         learning_rate=0.03, n_estimators=500, subsample=0.8, max_depth=3)
model.fit(X_train, y_train)

# 创建Streamlit界面
st.title("Achilles Tendon Stress Prediction")

# 显示应用说明
st.write("""
This application is designed to predict and identify personalized risk factors related to increased Achilles tendon stress during the start running phase. 
After entering your running posture information on the right, the model will predict your Achilles tendon stress and provide a report. 
If your tendon stress is high, you can actively adjust your running posture to reduce the stress and prevent running injuries.
""")

# 在侧边栏中添加用户输入
st.sidebar.header("Input Parameters")

# 各种特征的输入控件
ankle_angle = st.sidebar.slider("Ankle plantar/dorsiflexion angle", min_value=-100.0, max_value=100.0, value=0.0)
ankle_inversion = st.sidebar.slider("Ankle in/eversion angle", min_value=-100.0, max_value=100.0, value=0.0)
ankle_rotation_angle = st.sidebar.slider("Ankle in/external rotation angle", min_value=-100.0, max_value=100.0, value=0.0)
ankle_moment = st.sidebar.slider("Ankle plantar/dorsiflexion moment", min_value=-20.0, max_value=20.0, value=0.5)
ankle_inversion_moment = st.sidebar.slider("Ankle in/eversion moment", min_value=-20.0, max_value=20.0, value=0.5)
ankle_power = st.sidebar.slider("Ankle power", min_value=-200.0, max_value=200.0, value=5.0)
grf = st.sidebar.slider("A/P GRF", min_value=-200.0, max_value=200.0, value=50.0)
ml_grf = st.sidebar.slider("M/L GRF", min_value=-200.0, max_value=200.0, value=50.0)
hip_flex_angle = st.sidebar.slider("Hip flex/extension angle", min_value=-100.0, max_value=100.0, value=0.0)
hip_rotation_angle = st.sidebar.slider("Hip in/external rotation angle", min_value=-100.0, max_value=100.0, value=0.0)
hip_rotation_moment = st.sidebar.slider("Hip in/external rotation moment", min_value=-20.0, max_value=20.0, value=0.0)
knee_flex_angle = st.sidebar.slider("Knee flex/extension angle", min_value=-100.0, max_value=100.0, value=0.0)
knee_rotation_angle = st.sidebar.slider("Knee in/external rotation angle", min_value=-100.0, max_value=100.0, value=0.0)
pelvic_rotation = st.sidebar.slider("Ipsi/contralateral pelvis rotation", min_value=-100.0, max_value=100.0, value=0.0)
peroneus_emg = st.sidebar.slider("EMG activation for peroneus longus", min_value=0.0, max_value=1.0, value=0.5)

# 用户输入的特征值
user_input = np.array([[ankle_angle, ankle_inversion, ankle_rotation_angle, ankle_moment, ankle_inversion_moment, 
                        ankle_power, grf, ml_grf, hip_flex_angle, hip_rotation_angle, hip_rotation_moment, 
                        knee_flex_angle, knee_rotation_angle, pelvic_rotation, peroneus_emg]])

# 预测
predicted_stress = model.predict(user_input)

# 显示预测结果
st.write(f"Predicted Achilles Tendon Stress: {predicted_stress[0]:.2f}")
