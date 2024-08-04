import streamlit as st
import pandas as pd
import time
import numpy as np
# import sklearn
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
import pickle


def page_home():
    st.title("Welcome to My ML Application")
    file_ = open("ML_Gif.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

    #st.markdown('## Problem Statement of this ML application:')
    st.markdown("""## <span style="color:aqua">Problem Statement of this ML application:</span>""",unsafe_allow_html=True)
    st.markdown('The Copper Industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.')
    st.markdown('Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.')


def page_price():
    #creating 2 columns in streamlit page for getting input from user
    st.markdown("""# <span style="color:aqua">Predict Selling Price</span>""",unsafe_allow_html=True)
    st.markdown(" ")
    col1,col2 = st.columns(2)
    with col1:
        quantity_tons_pred=st.text_input('Enter  Quantity in Tons (Range: 0.1 - 1000000000)')
        customer_pred=st.text_input('Enter  CustomerId (Example: 30071586)')#getting a input from User
        country_pred=st.text_input('Enter  Country (Range: 25-113)')
        status_pred=st.text_input('Enter  Status as either "0" or "1", "WON" means "1" and "LOST" means "0"')
        item_type_pred=st.text_input('Enter  item_type (Range: 0-6)')
        application_pred=st.text_input('Enter  Application (Range: 2-87)')
        thickness_pred=st.text_input('Enter  Thickness (Range: 0.1 - 6)')
        width_pred=st.text_input('Enter  Width (Range: 700-1980)')
    with col2:
        product_ref_pred=st.text_input('Enter  Product_ref (Range: 611728-1722207579)')
        item_manu_year_pred=st.text_input('Enter  Manufactured Year (Eg.2020)')
        item_manu_month_pred=st.text_input('Enter  Manufactured Month (Range: 1-12)')
        item_manu_day_pred=st.text_input('Enter  Manufactured Day (Range: 1-31)')
        item_deli_year_pred=st.text_input('Enter  Delivery Year (Eg.2020)')
        item_deli_month_pred=st.text_input('Enter  Delivery Month (Range: 1-12)')
        diff_date_pred=st.text_input('Enter the Differnce between Delivery date and Manufactured date in days')
    
    #loading a knowledge file for ExtratreeRegression
    with open('knowlege_pkl_LinearReg', 'rb') as file:
        model_ExtratreeReg=pickle.load(file)

    if st.button('Predict Selling Price'):#button
                placeholder=st.empty()
                #forming a dataframe p with each input variables 
                p=pd.DataFrame([quantity_tons_pred,customer_pred,country_pred,status_pred,item_type_pred,application_pred,thickness_pred,width_pred,product_ref_pred,item_manu_year_pred,item_manu_month_pred,item_manu_day_pred,item_deli_year_pred,item_deli_month_pred,diff_date_pred])
                #converting each column's datatype to float
                p=p.astype(float)
                #applying scaling using pipeline function
                pipeline = Pipeline([
                ('std_scalar', StandardScaler())])
                y_pred = model_ExtratreeReg.predict(pipeline.fit_transform([[float(quantity_tons_pred),float(customer_pred),float(country_pred),float(status_pred),float(item_type_pred),float(application_pred),float(thickness_pred),float(width_pred),float(product_ref_pred),float(item_manu_year_pred),float(item_manu_month_pred),float(item_manu_day_pred),float(item_deli_year_pred),float(item_deli_month_pred),float(diff_date_pred)]]))
                y_pred=int(y_pred[0])
                # HTML and CSS for highlighted text
                highlighted_output = f"""
                <div style="background-color: black; padding: 10px;">
                    {y_pred}
                </div>
                """
                st.write("Predicted Selling Price:- ")
                st.markdown(highlighted_output, unsafe_allow_html=True)
                #st.write("Predicted Selling Price:- ",y_pred)
                time.sleep(30)


def page_status():
    #creating 2 columns in streamlit page for getting input from user
    st.markdown("""# <span style="color:aqua">Predict Status</span>""",unsafe_allow_html=True)
    st.markdown(" ")
    col1,col2 = st.columns(2)
    with col1:
        quantity_tons_pred=st.text_input('Enter  Quantity in Tons (Range: 0.1 - 1000000000)')
        customer_pred=st.text_input('Enter  CustomerId (Example: 30071586)')#getting a input from User
        country_pred=st.text_input('Enter  Country (Range: 25-113)')
        item_type_pred=st.text_input('Enter  item_type (Range: 0-6)')
        application_pred=st.text_input('Enter  Application (Range: 2-87)')
        thickness_pred=st.text_input('Enter  Thickness (Range: 0.18 - 6)')
        width_pred=st.text_input('Enter  Width (Range: 700-1980)')
        product_ref_pred=st.text_input('Enter  Product_ref (Range: 611728-1722207579)')
    with col2:
        item_manu_year_pred=st.text_input('Enter  Manufactured Year (Eg.2020)')
        item_manu_month_pred=st.text_input('Enter  Manufactured Month (Range: 1-12)')
        item_manu_day_pred=st.text_input('Enter  Manufactured Day (Range: 1-31)')
        item_deli_year_pred=st.text_input('Enter  Delivery Year (Eg.2020)')
        item_deli_month_pred=st.text_input('Enter  Delivery Month (Range: 1-12)')
        diff_date_pred=st.text_input('Enter the Differnce between Delivery date and Manufactured date in days')
        selling_price_pred=st.text_input('Enter the Selling Price (Range: 243-1379)')
    
    #loading a knowledge file for LogisticRegression
    with open('knowlege_pkl_LogisticReg', 'rb') as file:
        model_LogisticReg=pickle.load(file)

    if st.button('Predict Status'):#button
                placeholder=st.empty()
                #forming a dataframe p with each input variables 
                p=pd.DataFrame([quantity_tons_pred,customer_pred,country_pred,item_type_pred,application_pred,thickness_pred,width_pred,product_ref_pred,item_manu_year_pred,item_manu_month_pred,item_manu_day_pred,item_deli_year_pred,item_deli_month_pred,diff_date_pred,selling_price_pred])
                #converting each column's datatype to float
                p=p.astype(float)
                #applying scaling using pipeline function
                pipeline = Pipeline([
                ('std_scalar', StandardScaler())])
                #Adjusting threshold value to 0.2 to get better accuracy
                y_pred = np.where(model_LogisticReg.predict_proba(pipeline.fit_transform([[float(quantity_tons_pred),float(customer_pred),float(country_pred),float(item_type_pred),float(application_pred),float(thickness_pred),float(width_pred),float(product_ref_pred),float(item_manu_year_pred),float(item_manu_month_pred),float(item_manu_day_pred),float(item_deli_year_pred),float(item_deli_month_pred),float(diff_date_pred),float(selling_price_pred)]]))[:,1]>0.2,1,0)
                y_pred=int(y_pred[0])
                # HTML and CSS for highlighted text
                highlighted_output = f"""
                <div style="background-color: black; padding: 10px;">
                    {y_pred}
                </div>
                """
                st.write("Predicted Status \"1\" means \"WON\" and \"0\" means \"LOST\"  :- ")
                st.markdown(highlighted_output, unsafe_allow_html=True)
                time.sleep(30)


def main():
    st.sidebar.title("Page Navigation👇")
    selection = st.sidebar.selectbox("",["Home Page 🏠", "Predict Selling Price", "Predict Status"])

    if selection == "Home Page 🏠":
        page_home()
    elif selection == "Predict Selling Price":
        page_price()
    elif selection == "Predict Status":
        page_status()

if __name__=="__main__":
    main()
