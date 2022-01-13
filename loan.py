import pandas as pd
import streamlit as st
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
import plotly.express as px

loandata = pd.read_csv(r'credit_train.csv',index_col=[0])
loandata = loandata.drop(columns=['Customer ID'],axis=1)
loandata = loandata[loandata["Credit Score"]<2000]
loandata = loandata[loandata["Current Loan Amount"]<50000000]

loandata1 = loandata.copy()
loandata1["Months since last delinquent"] = loandata1["Months since last delinquent"].fillna(0)
loandata1["Home Ownership"] = loandata1["Home Ownership"].replace(['HaveMortgage'],'Home Mortgage')
loandata1 = loandata1.dropna()

le = preprocessing.LabelEncoder()
loandata1["Loan Status"] = le.fit_transform(loandata1["Loan Status"])
loandata1["Term"] = le.fit_transform(loandata1["Term"])
loandata1["Years in current job"] = le.fit_transform(loandata1["Years in current job"])
loandata1["Home Ownership"] = le.fit_transform(loandata1["Home Ownership"])
loandata1["Purpose"] = le.fit_transform(loandata1["Purpose"])

loandata1["Loan Status"] = loandata1["Loan Status"].astype(int)
sm = SMOTE(random_state = 4)
x = loandata1.drop(labels= 'Loan Status', axis = 1)
y = loandata1["Loan Status"]
x,y = sm.fit_resample(x,y)

x = x.drop(columns=['Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens', 'Purpose' ],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

XGBCV = XGBClassifier(eta=1.0,max_depth=10, n_estimators=20, random_state=4)
XGBCV.fit(X_train, y_train)

def loan():
    st.write("""# House Loan Approval Calculator""")
    st.sidebar.header('User Input Features')

    def user_input_features():
        CurrentLoanAmount = st.sidebar.number_input('Current Loan Amount', 0)
        Term = st.sidebar.slider('Long term: 0, Short term: 1',0,1)
        CreditScore = st.sidebar.number_input('Credit Score: 0-750', 0)
        AnnualIncome = st.sidebar.number_input('Current Annual Income', 0)
        Yearsincurrentjob = st.sidebar.slider('Years in current job:0 for 0 years, 10 for 10+ years', 0, 10)
        HomeOwnership = st.sidebar.slider('0: Home mortgage, 1: Own home 2: Rent', 0, 2)
        MonthlyDebt = st.sidebar.number_input('Current Monthly Debt',0)
        YearsofCreditHistory = st.sidebar.number_input('Years of Credit History', 0)
        Monthssincelastdelinquent = st.sidebar.number_input('Months since last delinquent', 0)
        NumberofOpenAccounts = st.sidebar.number_input('Number of Open Accounts', 0)

        data = {
            'Current Loan Amount' : CurrentLoanAmount,
            'Term' : Term,
            'Credit Score' : CreditScore,
            'Annual Income' : AnnualIncome,
            'Years in current job' : Yearsincurrentjob,
            'Home Ownership' : HomeOwnership,
            'Monthly Debt' : MonthlyDebt,
            'Years of Credit History' : YearsofCreditHistory,
            'Months since last delinquent' : Monthssincelastdelinquent,
            'Number of Open Accounts' : NumberofOpenAccounts
        }

        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

    st.subheader('User Input Features')
    st.write(input_df)

    prediction = XGBCV.predict(input_df)
    prediction_proba = XGBCV.predict_proba(input_df)

    st.subheader('Prediction for Loan Approval')
    if prediction == 0:
        st.write("No")
        st.write("Application should be reconsidered")
    elif prediction == 1:
        st.write("Yes")
        st.write("Application should be approved")

    st.subheader('Prediction Probability')
    prediction_proba_c = pd.DataFrame(prediction_proba)
    prediction_proba_c = prediction_proba_c.T
    st.write(prediction_proba)
    fig = px.pie(prediction_proba_c, values=prediction_proba_c.columns[0])
    st.plotly_chart(fig)

