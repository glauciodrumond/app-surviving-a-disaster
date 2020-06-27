# import libraries
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier


# function to load data
@st.cache
def get_data():
    return pd.read_csv('data.csv')


# function to train the model
def train_model():
    data = get_data()
    # create target and features
    X = data.drop('Survived', axis=1).values
    y = data['Survived'].values
    clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=1, gamma=0,
                        learning_rate=0.05, max_delta_step=0, max_depth=5,
                        min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                        nthread=None, objective='binary:logistic', random_state=42,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                        silent=None, subsample=1, verbosity=1)
    clf.fit(X, y)
    return clf


# create a data frame
data = get_data()

# train the model
model = train_model()

# App title
st.image("2210.jpg", width=700)
st.title("TITANIC - DATA APP")
st.subheader("Surviving a Disaster")
st.markdown('Welcome to this **Data Science** app. With the data gathered from the disaster '
            'with the Titanic,  using **Machine Learning**, it is possible to predict whether '
            'a passenger survives or not.  \n'
            'In a [Kaggle](https://www.kaggle.com/c/titanic/overview) competition, the '
            'algorithm used in this application predicted with '
            'almost 80% accuracy whether the passenger survived or not. So, try the app and '
            'see your fate, if you were in the Titanic.')


# data frame first 10 entries
st.subheader("Data")
st.markdown("This is how our database is structured. Here we have the first ten data frame entries.")
st.dataframe(data.head(10))

# collect the information to perform the prediction
# column Age
st.subheader("Age:")
Age = st.number_input("How old are you?", value=1)

# column Sex_Male
st.subheader("Gender:")
Sex_Male = st.selectbox("What is your gender?", ("Choose", "Male", "Female"))
Sex_male = 1 if Sex_Male == "Male" else 0

# columns SibSp
st.subheader("Travelling with:")
s = st.selectbox("Sibling or Spouse?", ("Choose", "Yes", "No"))
if s == "Yes":
    SibSp = st.number_input('How many?', value=data.SibSp.min())
else:
    SibSp = 0

# column Parch
p = st.selectbox("Parent or Child?", ("Choose", "Yes", "No"))
if p == "Yes":
    Parch = st.number_input('How many?', value=1)
else:
    Parch = 0

# column Family
Family = SibSp + Parch + 1

# column Pclass
st.subheader("Fare and Class")
st.write(
    "* Third class ticket cost around £7 in 1912 which is nearly £800 in today's money.  \n"
    "* Second class ticket cost around £13 or nearly £1500 today money.  \n"
    "* First class ticket would have set you back a minimum of £30 or more than £3300 today.")

c = st.selectbox("Choose a class", ("Class", "First", "Second", "Third"))
if c == "First":
    Pclass_2 = 0
    Pclass_3 = 0
    Fare = 35.0

if c == "Second":
    Pclass_2 = 1
    Pclass_3 = 0
    Fare = 13.0

if c == "Third":
    Pclass_2 = 0
    Pclass_3 = 1
    Fare = 7.0

else:
    st.markdown("")

# column embarked
st.subheader("Boarding city")
st.write(
    "There were three boarding points for the Titanic:  \n"
    "1. Southampton is a city in Hampshire, South East England   \n"
    "2. Cherbourg-Octeville is a city situated at the northwestern France   \n"
    "3. Queenstown, is a tourist seaport town on the south coast of County Cork, Ireland  \n"
    "  \n"
    "**Choose one:**")

e = st.selectbox("", ("City", "Cherbourg", "Queenstown", "Southampton"))
if e == "Cherbourg":
    Embarked_Q = 0
    Embarked_S = 0

if e == "Queenstown":
    Embarked_Q = 1
    Embarked_S = 0

if e == "Southampton":
    Embarked_Q = 0
    Embarked_S = 1

else:
    st.markdown("")

# add button in screen
btn_predict = st.button("PREDICT")

# check if button was pressed
if btn_predict:
    X_predict = [Age, SibSp, Parch, Fare, Family, Sex_male,
                 Embarked_Q, Embarked_S, Pclass_2, Pclass_3]
    X_predict = np.array(X_predict).reshape((1, -1))
    result = model.predict(X_predict)

# print result
    st.subheader("According to the data provided...")
    if result == 1:
        st.header("Survived the Titanic disaster")
        st.image('2309.jpg', width=350)
    else:
        st.header("The news is not good ..... You died...")
        st.image('2519.jpg', width=350)


st.subheader('Contact me on my '
             '[LinkedIn](https://www.linkedin.com/in/glaucio-drumond-1734a018b/)')
st.subheader('*Created by*: ***Glaucio Drumond***')
st.write('Images: [Freepik](http://www.freepik.com) - Designed by vectorpouchs')
