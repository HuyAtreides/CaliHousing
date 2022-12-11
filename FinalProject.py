import streamlit as st
import pandas as pd
import numpy as np
import joblib #to load a saved model
import base64  #to open .gif files in streamlit app
import os

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

baseData = pd.read_csv("housing.csv")

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def Create_pipeline(model=None):
    housing = pd.read_csv("housing.csv")
    # Them column income_cat dung de chia data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Chia xong thi delete column income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    return full_pipeline

def transformData(val):
    full_pipline = Create_pipeline()
    return full_pipline.fit_transform(val)

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def display_scores(scores):
    st.text("Trung bình: %.2f" % (scores.mean()))
    st.text("Độ lệch chuẩn: %.2f" % (scores.std()))

def RunModel(model=None):
    housing = baseData
    numberOfTest = st.slider("Chọn số lượng có trong dữ liệu mẫu để kiểm tra theo %", 10, 50, 20, 1)
    numberOfTest = int(len(housing)*(numberOfTest/100))
    sample = housing.sample(numberOfTest)

    #Chuẩn bị dữ liệu
    st.markdown("Dữ liệu sample để dự đoán")
    st.write(sample.drop(["median_house_value"], axis=1).reset_index(drop=True).head(10))
    st.markdown("Tổng số dữ liệu mẫu được chọn: %d" % len(sample))
    labels = sample["median_house_value"].reset_index(drop=True)
    dt = pd.concat([sample, housing])
    sample_prepared = transformData(dt)

    #Sau khi dự đoán
    sample_prepared = model.predict(sample_prepared[:numberOfTest])
    predict_data = pd.DataFrame(sample_prepared, columns=["Predict value"])

    st.markdown("Kết quả dự đoán so với dữ liệu thực tế")
    st.write(pd.concat([predict_data, labels], axis=1))

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.title("Đánh giá kết quả")
    fig, ax = plt.subplots()
    plt.xlabel("Giá trị dự đoán")
    plt.ylabel("Giá trị thực tế")
    ax.scatter(predict_data, labels)
    st.pyplot(fig)

    mse_test = mean_squared_error(labels, predict_data)
    rmse_test = np.sqrt(mse_test)
    st.text('Độ lệch chuẩn bình phương - test: %.2f' % rmse_test)

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

def PredictSingleValue(val, model)->float:
    result = 0.0
    dt = pd.concat([val, baseData])
    sample_prepared = transformData(dt)
    result = model.predict(sample_prepared[:1])
    return result

def input_feature():
    st.subheader('Vui lòng nhập các thông tin để dự đoán giá nhà.')
    st.sidebar.header("Cung cấp thông tin đi:")
    
    #Information
    longitude = st.sidebar.number_input("Kinh độ", value=-122.5000, format="%.3f")
    latitude = st.sidebar.number_input("Vĩ độ", value = 37.9200, format="%.3f")
    housing_median_age = st.sidebar.number_input("Trung bình tuổi thọ của nhà trong khu vực", min_value=0, value=20, step=1)
    total_rooms = st.sidebar.number_input("Số phòng", min_value=0, value=1000, step=100, format="%d")
    total_bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=0, value=500, step=10, format="%d")
    population = st.sidebar.number_input("Số người dân", min_value=0, value=1000, step=100, format="%d")
    households = st.sidebar.number_input("Số hộ gia đình", min_value=0, value=500, step=50, format="%d")
    median_income = st.sidebar.number_input("Bình quân thu nhập", min_value=0, value = 1000, format="%d")
    ocean_proximity = st.sidebar.radio("Vị trí so với biển gần nhất", options=['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

    data = (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)

    return data

def Show(model_src):
    model = joblib.load(model_src)
    RunModel(model)
    data = input_feature()
    data = pd.DataFrame([data], columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"])
    st.write(data)
    if st.button("Dự đoán"):
        st.write("Dự đoán nhà của bạn có giá tiền khoảng: %.2f" % PredictSingleValue(data, model))


app_mode = st.sidebar.selectbox('Chọn Trang', 
                                ['Trang chủ', 
                                'Linear Regression', 
                                'Decision Tree', 
                                'Random Forest Regression',
                                'Random Forest Regression Grid Search CV',
                                'Random Forest Regression Grid Random CV'
                                ]) 

# Chương trình chính
if app_mode=='Trang chủ':
    st.title('Dự đoán giá nhà:')  
    st.markdown('Dữ liệu đầu vào:')
    data = pd.read_csv('housing.csv')
    st.write(data.head(20))

elif app_mode=='Linear Regression':
    st.title("Dự đoán giá nhà ở Cali bằng phương pháp Hồi quy tuyến tính")
    Show("model_lin_reg.pkl")

elif app_mode=="Decision Tree":
    st.title("Dự đoán giá nhà ở Cali bằng phương pháp Decision tree")
    Show("tree_reg.pkl")

elif app_mode=='Random Forest Regression':
    st.title("Dự đoán giá nhà ở Cali bằng phương pháp Random Forest Regression")
    Show("random_forest_regression.pkl")

elif app_mode=='Random Forest Regression Grid Search CV':
    st.title("Dự Đoán giá nhà ở Cali bằng phương pháp Random Forest Regression Grid Search CV")
    Show('random_forest_regression_grid_search_cv_model.pkl')

elif app_mode=='Random Forest Regression Grid Random CV':
    st.title("Dự Đoán giá nhà ở Cali bằng phương pháp Random Forest Regression Random Search CV")
    Show('random_forest_regression_random_search_cv_model.pkl')
