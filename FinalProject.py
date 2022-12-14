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
    st.text("Trung b??nh: %.2f" % (scores.mean()))
    st.text("????? l???ch chu???n: %.2f" % (scores.std()))

def RunModel(model=None):
    housing = baseData
    numberOfTest = st.slider("Ch???n s??? l?????ng c?? trong d??? li???u m???u ????? ki???m tra theo %", 10, 50, 20, 1)
    numberOfTest = int(len(housing)*(numberOfTest/100))
    sample = housing.sample(numberOfTest)

    #Chu???n b??? d??? li???u
    st.markdown("D??? li???u sample ????? d??? ??o??n")
    st.write(sample.drop(["median_house_value"], axis=1).reset_index(drop=True).head(10))
    st.markdown("T???ng s??? d??? li???u m???u ???????c ch???n: %d" % len(sample))
    labels = sample["median_house_value"].reset_index(drop=True)
    dt = pd.concat([sample, housing])
    sample_prepared = transformData(dt)

    #Sau khi d??? ??o??n
    sample_prepared = model.predict(sample_prepared[:numberOfTest])
    predict_data = pd.DataFrame(sample_prepared, columns=["Predict value"])

    st.markdown("K???t qu??? d??? ??o??n so v???i d??? li???u th???c t???")
    st.write(pd.concat([predict_data, labels], axis=1))

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.title("????nh gi?? k???t qu???")
    fig, ax = plt.subplots()
    plt.xlabel("Gi?? tr??? d??? ??o??n")
    plt.ylabel("Gi?? tr??? th???c t???")
    ax.scatter(predict_data, labels)
    st.pyplot(fig)

    mse_test = mean_squared_error(labels, predict_data)
    rmse_test = np.sqrt(mse_test)
    st.text('????? l???ch chu???n b??nh ph????ng - test: %.2f' % rmse_test)

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

def PredictSingleValue(val, model)->float:
    result = 0.0
    dt = pd.concat([val, baseData])
    sample_prepared = transformData(dt)
    result = model.predict(sample_prepared[:1])
    return result

def input_feature():
    st.subheader('Vui l??ng nh???p c??c th??ng tin ????? d??? ??o??n gi?? nh??.')
    st.sidebar.header("Cung c???p th??ng tin ??i:")
    
    #Information
    longitude = st.sidebar.number_input("Kinh ?????", value=-122.5000, format="%.3f")
    latitude = st.sidebar.number_input("V?? ?????", value = 37.9200, format="%.3f")
    housing_median_age = st.sidebar.number_input("Trung b??nh tu???i th??? c???a nh?? trong khu v???c", min_value=0, value=20, step=1)
    total_rooms = st.sidebar.number_input("S??? ph??ng", min_value=0, value=1000, step=100, format="%d")
    total_bedrooms = st.sidebar.number_input("S??? ph??ng ng???", min_value=0, value=500, step=10, format="%d")
    population = st.sidebar.number_input("S??? ng?????i d??n", min_value=0, value=1000, step=100, format="%d")
    households = st.sidebar.number_input("S??? h??? gia ????nh", min_value=0, value=500, step=50, format="%d")
    median_income = st.sidebar.number_input("B??nh qu??n thu nh???p", min_value=0, value = 1000, format="%d")
    ocean_proximity = st.sidebar.radio("V??? tr?? so v???i bi???n g???n nh???t", options=['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

    data = (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)

    return data

def Show(model_src):
    model = joblib.load(model_src)
    RunModel(model)
    data = input_feature()
    data = pd.DataFrame([data], columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"])
    st.write(data)
    if st.button("D??? ??o??n"):
        st.write("D??? ??o??n nh?? c???a b???n c?? gi?? ti???n kho???ng: %.2f" % PredictSingleValue(data, model))


app_mode = st.sidebar.selectbox('Ch???n Trang', 
                                ['Trang ch???', 
                                'Linear Regression', 
                                'Decision Tree', 
                                'Random Forest Regression',
                                'Random Forest Regression Grid Search CV',
                                'Random Forest Regression Grid Random CV'
                                ]) 

# Ch????ng tr??nh ch??nh
if app_mode=='Trang ch???':
    st.title('D??? ??o??n gi?? nh??:')  
    st.markdown('D??? li???u ?????u v??o:')
    data = pd.read_csv('housing.csv')
    st.write(data.head(20))

elif app_mode=='Linear Regression':
    st.title("D??? ??o??n gi?? nh?? ??? Cali b???ng ph????ng ph??p H???i quy tuy???n t??nh")
    Show("model_lin_reg.pkl")

elif app_mode=="Decision Tree":
    st.title("D??? ??o??n gi?? nh?? ??? Cali b???ng ph????ng ph??p Decision tree")
    Show("tree_reg.pkl")

elif app_mode=='Random Forest Regression':
    st.title("D??? ??o??n gi?? nh?? ??? Cali b???ng ph????ng ph??p Random Forest Regression")
    Show("random_forest_regression.pkl")

elif app_mode=='Random Forest Regression Grid Search CV':
    st.title("D??? ??o??n gi?? nh?? ??? Cali b???ng ph????ng ph??p Random Forest Regression Grid Search CV")
    Show('random_forest_regression_grid_search_cv_model.pkl')

elif app_mode=='Random Forest Regression Grid Random CV':
    st.title("D??? ??o??n gi?? nh?? ??? Cali b???ng ph????ng ph??p Random Forest Regression Random Search CV")
    Show('random_forest_regression_random_search_cv_model.pkl')
