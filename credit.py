import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import plotly.express as px

train_data = pd.read_csv(r'cs-training.csv', index_col=[0])
train = train_data.drop(train_data[train_data['DebtRatio'] > 5].index)
train = train.fillna(train.median())
train = train.drop(train[train['RevolvingUtilizationOfUnsecuredLines'] > 10].index)

train_set = train.copy()
train_set.loc[train_set['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
train_set.loc[train_set['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
train_set.loc[train_set['NumberOfTimes90DaysLate'] > 90] = 18

train_set["SeriousDlqin2yrs"] = train_set["SeriousDlqin2yrs"].astype(int)
train_set["RevolvingUtilizationOfUnsecuredLines"] = train_set["RevolvingUtilizationOfUnsecuredLines"].astype("float32")
train_set["age"] = train_set["age"].astype(int)
train_set["NumberOfTime30-59DaysPastDueNotWorse"] = train_set["NumberOfTime30-59DaysPastDueNotWorse"].astype(int)
train_set["DebtRatio"] = train_set["DebtRatio"].astype("float32")
train_set["MonthlyIncome"] = train_set["MonthlyIncome"].astype(int)
train_set["NumberOfOpenCreditLinesAndLoans"] = train_set["NumberOfOpenCreditLinesAndLoans"].astype(int)
train_set["NumberOfTimes90DaysLate"] = train_set["NumberOfTimes90DaysLate"].astype(int)
train_set["NumberRealEstateLoansOrLines"] = train_set["NumberRealEstateLoansOrLines"].astype(int)
train_set["NumberOfTime60-89DaysPastDueNotWorse"] = train_set["NumberOfTime60-89DaysPastDueNotWorse"].astype(int)
train_set["NumberOfDependents"] = train_set["NumberOfDependents"].astype(int)

train_set = train_set[train_set['SeriousDlqin2yrs'] < 2]

X = train_set.drop('SeriousDlqin2yrs', axis=1)
y = train_set['SeriousDlqin2yrs']
sm = SMOTE(random_state=4)
X, y = sm.fit_resample(X, y)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
X = X.drop(["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

XGBCV = XGBClassifier(eta=1.0,max_depth=10, n_estimators=20, random_state=4)
XGBCV.fit(X_train, y_train)

def credit():
    st.write("""# Credit Card Approval Calculator""")
    st.sidebar.header('User Input Features')


    def user_input_features():
        age = st.sidebar.slider('Age', 0)
        MonthlyIncome = st.sidebar.number_input('Current monthly income', 1)
        DebtRatio = st.sidebar.number_input('Monthly expenses divided by monthly gross income', 0.000001)
        NumberOfDependents = st.sidebar.slider('Number of dependents', 0, 20)
        RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input('Total balance on credit cards and personal lines of credit', 0.001)
        NumberOfTime30to59DaysPastDueNotWorse = st.sidebar.slider('Number of times 1 to 2 months late debt payment in the last 2 years', 0, 20)
        NumberOfTime60to89DaysPastDueNotWorse = st.sidebar.slider('Number of times 2 to 3 months late debt payment in the last 2 years',0, 20)
        NumberOfTimes90DaysLate = st.sidebar.slider('Number of times late debt payment over 3 months', 0, 20)


        data = {'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
                'age': age,
                'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30to59DaysPastDueNotWorse,
                'DebtRatio': DebtRatio,
                'MonthlyIncome': MonthlyIncome,
                'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
                'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60to89DaysPastDueNotWorse,
                'NumberOfDependents': NumberOfDependents}
        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

    st.subheader('User Input Features')
    st.write(input_df)

    prediction = XGBCV.predict(input_df)
    prediction_proba = XGBCV.predict_proba(input_df)

    st.subheader('Prediction for Serious Deliquency in 2 years')
    if prediction == 0:
        st.write("No")
        st.write("Application should be approved")
    elif prediction == 1:
        st.write("Yes")
        st.write("Application should be reconsidered")

    st.subheader('Prediction Probability')

    prediction_proba_c = pd.DataFrame(prediction_proba)
    prediction_proba_c = prediction_proba_c.T
    st.write(prediction_proba)
    fig = px.pie(prediction_proba_c, values=prediction_proba_c.columns[0])
    st.plotly_chart(fig)

