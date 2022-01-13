#main
import streamlit as st
import credit
import loan

def main():
    page = st.sidebar.selectbox("Choose a page", ["Credit Card", "Loan"])

    if page == "Credit Card":
        credit.credit()
    elif page == "Loan":
        loan.loan()
main()