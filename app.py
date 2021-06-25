from pycaret.regression import load_model, predict_model
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import datetime,time
import streamlit.components.v1 as stc

model = load_model('AI_invasion_regression_model_25_06_2021')
# model = load_model('AI_invasion_classification_model_25_06_2021')
def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

# def predict(model, input_df):
#     predictions_df = predict_model(estimator=model, data=input_df)
#     predictions = predictions_df['Label'][0]
#     if predictions[0] == 0:
#         prediction = 'not-purchased'
#     else:
#         predictions[0] == 1
#         prediction = 'purchased'
#     return predictions

def run():

    # from PIL import Image
    # image = Image.open('Images\logo.png')
    # image_sidebar = Image.open('renager.jpg')

    # st.image(image,use_column_width=False)
    # st.sidebar.image(image_sidebar)  
    add_selectbox = st.sidebar.selectbox("Please select the option of your choice",("Regression model","About"))

    st.sidebar.info("")
    
    
    if add_selectbox == 'Regression model':
        stc.html("""
		<div style="background-color:#31333F;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Regression model</h1>
		</div>	""")
        
        with st.form(key='mlform'):
            total_bill=st.slider('Total Bil',1,6000)
            gender= st.selectbox('Gender', ['Male','Female'])
            time= st.selectbox('Time', ['Dinner', 'Lunch'])
            size= st.slider('Size',1,5)
            day= st.selectbox('Day', ['Sun', 'Wed', 'Sat', 'Tues', 'Thur', 'Mon', 'Fri'])
            smoker= st.selectbox('smoker', ['Yes', 'No'])
            
            output=""
            input_dict = {
                        'total_bill': total_bill,
                        'gender': gender,
                        'time':time,
                        'size':size,
                        'day':day,
                        'smoker':smoker
                        }
            input_df = pd.DataFrame([input_dict])
            submit_message = st.form_submit_button(label='Predict Tip')
            
        if submit_message:
            output = predict(model= model, input_df=input_df)
            output = '₦' + str(output)
            st.success('The predicted Price is: {}'.format(output))


    
             
if __name__ == '__main__':
    run()