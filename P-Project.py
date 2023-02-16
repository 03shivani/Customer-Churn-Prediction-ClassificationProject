import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
import us
from streamlit_option_menu import option_menu

#Set Page Layout
st.set_page_config(layout="wide")


#pip install streamlit-option-menu
with st.sidebar:
    selected = option_menu(None, ["EDA", "Prediction"],
                           styles={
            "nav-link-selected": {"background-color": "#262730"}        ##262730
        },
    #icons=['house', 'cloud-upload', None, "list-task", 'gear'],
                           menu_icon="cast", default_index=1)



# title
st.markdown("<h1 style='text-align: center;'>Churn prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>=====================================================================</h3>", unsafe_allow_html=True)


# creating function for null value imputation
def nullvalueimputer(data,x,y):
    data[y].replace(['Nan','NAN','NaN','nan'],np.nan,inplace=True)
    #separating testing data
    test_data = data[data[y].isin([np.nan])]
    x_test = pd.DataFrame([test_data[x]]).T
    y_test = pd.DataFrame([test_data[y]]).T

    #separating training data
    train_data = data.dropna(axis = 0)
    y_train = pd.DataFrame(train_data[y])
    x_train = pd.DataFrame(train_data[x])

    #creating regression model
    reg = LinearRegression()
    reg.fit(x_train,y_train)

    #making predictions
    y_pred = reg.predict(x_test)

    #imputing predicted values
    test_data.loc[test_data[y].isnull(), y] = y_pred

    #concating and creating a fresh dataframe
    new_df = pd.concat([test_data, train_data])
    new_df = new_df.sort_index(axis = 0)
    return new_df


# import csv file
#@st.cache(allow_output_mutation=True)
def get_data():
    df = pd.read_csv("C:\\Users\\win7\\OneDrive\\Desktop\\Churn.csv")
    df = df.drop(['Unnamed: 0'],axis=1)
    df.columns = df.columns.str.replace('.', '_')
    #df['Churn'] = np.where(df['Churn']=='FALSE',0,1)
    return df
data = get_data()

#null value imputation for day_charge
new = nullvalueimputer(data=data,x="day_mins",y="day_charge")
#null value imputation for eve_mins
df = nullvalueimputer(new,"eve_charge","eve_mins")


states = us.states.mapping("abbr","name")
State = df['state'].map(lambda x: states.get(x, x))

df.insert(loc=0, column='State_names', value=State)
df.rename(columns = {'state':'State_codes'}, inplace = True)


###########################################################################################################################################
 












if selected == 'EDA':
    ## DataSet
    st.sidebar.markdown("<h3 style='text-align: center;'>Dataset</h3>", unsafe_allow_html=True)
    Data_view = st.sidebar.checkbox('Dataset')
    if Data_view:
        st.markdown("<h3 style='text-align: center;'>Dataset</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(5))
        st.write(df.shape)


    categorical = ['State_names', 'area_code', 'voice_plan', 'intl_plan']#
    numeric = ['account_length', 'voice_messages', 'intl_mins', 'intl_calls', 'intl_charge',
               'day_charge', 'eve_mins', 'day_mins', 'day_calls', 'eve_calls', 'eve_charge',
               'night_mins', 'night_calls', 'night_charge', 'customer_calls']


    ##EDA
    st.sidebar.markdown("<h2 style='text-align: center;'>----------------------------</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center;'>EDA</h3>", unsafe_allow_html=True)

    EDA = st.sidebar.checkbox('EDA')
    if EDA:
        Report = st.sidebar.selectbox("Select Report",("Descriptive Statistics","Univeriate","Baiveriate","Multivariate"))
        if Report == "Descriptive Statistics":
            st.markdown("<h3 style='text-align: center;'>Describe</h3>", unsafe_allow_html=True)
            st.table(df.describe())
            st.markdown("<h3 style='text-align: center;'>Correlation</h3>", unsafe_allow_html=True)
            matrix = df.corr()
            matrix = round(matrix,2)
            fig = px.imshow(matrix, width=800, height=800, text_auto=True)
            st.plotly_chart(fig)

        if Report == "Univeriate":
            selected_col = st.sidebar.selectbox("Select Type",df.columns)
            if selected_col in categorical:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                                    subplot_titles=["Bar Chart", "Pie Chart"], column_widths=[0.6, 0.4])
                fig.add_trace(go.Bar(x=df[selected_col].value_counts().index, y=df[selected_col].value_counts()), row=1, col=1)
                fig.add_trace(go.Pie(labels=df[selected_col].value_counts().index, values=df[selected_col].value_counts()), row=1, col=2)
                fig.update_layout(
                    {'title': {'text': f"plots for {selected_col}", 'x': 0.5, 'y': 0.9, 'font_size': 30, 'font_color': '#FFD700'}},
                height = 600, width = 900, showlegend = False)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig)


                #for i in categorical:
                #    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                #                        subplot_titles=["Bar Chart", "Pie Chart"], column_widths=[0.6, 0.4])
                #    fig.add_trace(go.Bar(x=df[i].value_counts().index, y=df[i].value_counts()), row=1, col=1)
                #    fig.add_trace(go.Pie(labels=df[i].value_counts().index, values=df[i].value_counts()), row=1, col=2)
                #    fig.update_layout(
                #        {'title': {'text': f"plots for {i}", 'x': 0.5, 'y': 0.9, 'font_size': 25, 'font_color': 'Blue'}},
                #    height = 600, width = 900, showlegend = False)
                #    fig.update_xaxes(tickangle=-45)
                #    st.plotly_chart(fig)
            else:
                p1 = go.Box(y=df[selected_col], marker=dict(color='Orange'), name=f'{selected_col}')
                p2 = go.Histogram(x=df[selected_col], marker=dict(color='yellowgreen'))
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "Histogram"}]],
                                    subplot_titles=["Box Plot", "Histogram"])

                fig.append_trace(p1, row=1, col=1)
                fig.append_trace(p2, row=1, col=2)
                fig.update_layout(
                    {'title': {'text': f"plots for {selected_col}", 'x': 0.5, 'y': 0.9, 'font_size': 30, 'font_color': '#FFD700'}}
                    , height=500, width=900, showlegend=False)
                # fig.show()
                st.plotly_chart(fig)

                #for i in numeric:
                #    p1 = go.Box(y=df[i], marker=dict(color='Orange'), name=f'{i}')
                #    p2 = go.Histogram(x=df[i], marker=dict(color='yellowgreen'))
    #
                #    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "Histogram"}]],
                #                        subplot_titles=["Box Plot", "Histogram"])
    #
                #    fig.append_trace(p1, row=1, col=1)
                #    fig.append_trace(p2, row=1, col=2)
                #    fig.update_layout(
                #        {'title': {'text': f"plots for {i}", 'x': 0.5, 'y': 0.9, 'font_size': 25, 'font_color': 'Blue'}}
                #        , height=600, width=900, showlegend=False)
                #    #fig.show()
                #    st.plotly_chart(fig)

        if Report == "Baiveriate":
            selected_col = st.sidebar.selectbox("Select Column", df.columns)
            cl1, cl2 = st.columns([1,1])
            colo = ['#012D9C', '#7097FF']
            if selected_col in numeric:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "Box"}, {"type": "Box"}]],
                                subplot_titles=["Box Plot", "Histogram"], column_widths=[0.6, 0.4])

                churn = ['yes', 'no']
                color = ['#079FEB', '#550A35']

                for d, i in enumerate(churn):
                    fig.add_trace(
                        go.Violin(x=df['churn'][df['churn'] == i], y=df[selected_col][df['churn'] == i], name=i, legendgroup=i,
                                  showlegend=False, box_visible=True, meanline_visible=True, line_color=color[d]), row=1, col=1)
                for d, i in enumerate(churn):
                    fig.add_trace(go.Histogram(x=df[selected_col][df['churn'] == i], marker=dict(color=color[d]), name=i), row=1,
                                  col=2)

                fig.update_layout(barmode='stack', height=600, width=900)

                st.plotly_chart(fig)

            #                                 ####   Funnel Chart   ####
            else:
                pivot_table = df.groupby([selected_col, 'churn']).size().reset_index(name='count')
                pivot_table = pivot_table.pivot(index=selected_col, columns='churn', values='count').fillna(0)
                pivot_table = pivot_table.reset_index().rename_axis(None, axis=1)
                pivot_table.columns = [selected_col, 'No', 'Yes']

                df8 = pivot_table.melt(id_vars=selected_col, value_vars=['No', 'Yes'], value_name='count')
                fig1 = px.funnel(df8, x='variable', y='count', color=selected_col
                                          ,color_discrete_sequence=['#DB6574', '#03DAC5']
                                          ,title=selected_col)
                fig1.update_traces(textposition='auto', textfont=dict(color='#fff'))
                fig1.update_layout(autosize=True,
                                          margin=dict(t=110, b=50, l=70, r=40),
                                          xaxis_title=' ', yaxis_title=" ",
                                          plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                          title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif"),
                                          font=dict(color='#8a8d93',size=22)
                                         # ,legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                         , height = 600, width = 900)
                st.plotly_chart(fig1)

                   #                                 ####   Tree Map   ####

                pivot_table = df.groupby([selected_col, 'churn']).size().reset_index(name='count')
                pivot_table = pivot_table.pivot(index=selected_col, columns='churn', values='count').fillna(0)
                pivot_table = pivot_table.reset_index().rename_axis(None, axis=1)
                pivot_table.columns = [selected_col, 'No', 'Yes']

                df7 = pd.melt(pivot_table, id_vars=selected_col, value_vars=['No', 'Yes'])
                fig2 = px.treemap(df7, path=[selected_col, 'variable'], values='value')
                fig2.update_layout(height = 600, width = 900)
                st.plotly_chart(fig2)





        if Report == "Multivariate":
            tab1, tab2, tab3 =st.tabs(['Scatter','Map','New pie'])
            with tab1:
                cc1, cc2 = st.columns([1, 1])
                x_axis = cc1.selectbox("Select X_axis", numeric)
                Y_axis = cc2.selectbox("Select Y_axis", numeric)
                co1, co2, co3 = st.columns([1,1,1])
                uni2 = df['area_code'].unique()
                uni3 = df['voice_plan'].unique()
                uni4 = df['intl_plan'].unique()
                uni2 = np.insert(uni2, 0, "All")
                uni3 = np.insert(uni3, 0, "All")
                uni4 = np.insert(uni4, 0, "All")

                f2 = co1.selectbox("Area_code", uni2)
                f3 = co2.selectbox("Voice_plan", uni3)
                f4 = co3.selectbox("intl_plan", uni4)

                if f2 == "All":
                    df1 = df
                else:
                    df1 = df[df['area_code'] == f2]
                if f3 == "All":
                    df1 = df1
                else:
                    df1 = df1[df1['voice_plan'] == f3]
                if f4 == "All":
                    df1 = df1
                else:
                    df1 = df1[df1['intl_plan'] == f4]

                fig = px.scatter(df1, x=x_axis, y=Y_axis,
                                 color="churn",width=900,height=500)#hover_name="country", log_x=True,
                st.write("Buble chart")
                st.plotly_chart(fig)

            with tab2:
                cl1, cl2, cl3, cl4 = st.columns([1, 1, 1, 1])
                selected_col = cl1.selectbox("Select Value",numeric)
                f6 = cl2.selectbox("Area_code_", uni2)
                f7 = cl3.selectbox("Voice_plan_", uni3)
                f8 = cl4.selectbox("intl_plan_", uni4)

                if f6 == "All":
                    df2 = df
                else:
                    df2 = df[df['area_code'] == f6]
                if f7 == "All":
                    df2 = df2
                else:
                    df2 = df2[df2['voice_plan'] == f7]
                if f8 == "All":
                    df2 = df2
                else:
                    df2 = df2[df2['intl_plan'] == f8]
                table = pd.pivot_table(df2, values=selected_col, index=['State_codes'],
                                       columns=['churn'], aggfunc=np.sum)
                st.dataframe(df2)
                df2 = df2.groupby(['State_codes',selected_col]).size().reset_index(name='count')
                #   st.dataframe(df)
                fig4 = px.choropleth(df2, locations='State_codes', color='count', locationmode= "USA-states",
                               color_continuous_scale="PuBu", width=900, height=500)
                fig4.update_geos(fitbounds="locations", visible=False)
                fig4.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'),margin={"r":0,"t":100,"l":0,"b":0},paper_bgcolor='#0E1117',plot_bgcolor='#0E1117')
                st.plotly_chart(fig4)

            with tab3:
                grouped = df.groupby(['area_code', 'voice_plan', 'intl_plan'])['churn'].value_counts().reset_index(
                    name='count')

                # Create the sunburst chart
                fig = px.sunburst(grouped, path=['area_code', 'voice_plan', 'intl_plan', 'churn'], values='count',
                                  color_discrete_sequence=['#416768', '#439EA0'])

                fig.update_layout(autosize=True,
                                  margin=dict(t=110, b=50, l=70, r=40),
                                  xaxis_title=' ', yaxis_title=" ",
                                  plot_bgcolor='#1BB3D8', paper_bgcolor='#2d3035',
                                  title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif"),
                                  font=dict(color='#8a8d93'),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                  )

                # Show the plot
                st.plotly_chart(fig)



    









#######################################################################################################################################


import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing , linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing , linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(handle_unknown="ignore", sparse=False)
import base64

##############################################

#             Model

### catboost

from catboost import CatBoostClassifier
data1=['High', 'Low-Medium', 'Low', 'Medium-High', 'intl_plan_no',
       'intl_plan_yes', 'voice_plan_yes', 'voice_plan_no', 'area_code_415',
       'area_code_408', 'area_code_510', 'account_length', 'voice_messages',
       'intl_mins', 'intl_calls', 'intl_charge', 'day_mins', 'day_calls',
       'day_charge', 'eve_mins', 'eve_calls', 'eve_charge', 'night_mins',
       'night_calls', 'night_charge', 'customer_calls']
sc=pd.read_csv("C:\\Users\\win7\\Downloads\\state-category.csv")
unique_states=pd.read_csv("C:\\Users\\win7\\Downloads\\unique_states.csv")
import pickle
model = pickle.load(open('C:/Users/win7/Downloads/ch.pkl','rb'))
##############################   User Input    
    
if selected == 'Prediction': 
    #st.sidebar.markdown("<h1 style='text-align: left;'>Insert</h1>", unsafe_allow_html=True)
    option = st.sidebar.radio('Insert',('Values','Dataset'))
    if option == 'Values':
        def user_input_features():
            st.write("**Please provide the following inputs:**")
            col1, col2 = st.columns(2)

            with col1:
                state_options = unique_states['State_code_name'].tolist()
                state = st.selectbox('State', options=state_options)
                area_code = st.selectbox('Area Code',('area_code_408','area_code_415','area_code_510'))
                account_length = st.number_input('Account Length', value=0)
                voice_plan = st.selectbox('Voice Plan ',('Yes','No'))
                voice_messages = st.number_input('Voice Messages', value=0)
                intl_plan = st.selectbox('Intl Plan',('Yes','No'))
                intl_mins = st.number_input('Intl Mins', value=0.0, step=0.1)
                intl_calls = st.number_input('Intl Calls', value=0)
                intl_charge = st.number_input('Intl Charge', value=0.0, step=0.1)
                day_mins = st.number_input('Day Mins', value=0.0, step=0.1)

            with col2:
                day_calls = st.number_input('Day Calls', value=0)
                day_charge = st.number_input('Day Charge', value=0.0, step=0.1)
                eve_mins = st.number_input('Eve Mins', value=0.0, step=0.1)
                eve_calls = st.number_input('Eve Calls', value=0)
                eve_charge = st.number_input('Eve Charge', value=0.0, step=0.1)
                night_mins = st.number_input('Night Mins', value=0.0, step=0.1)
                night_calls = st.number_input('Night Calls', value=0)
                night_charge = st.number_input('Night Charge', value=0.0, step=0.1)
                customer_calls = st.number_input('Customer Calls', value=0)
      
            data = {'state':state,
                   'area_code':area_code,
                   'account_length':account_length,
                   'voice_plan':voice_plan,
                   'voice_messages':voice_messages,
                   'intl_plan':intl_plan,
                   'intl_mins':intl_mins,
                   'intl_calls':intl_calls,
                   'intl_charge':intl_charge,
                   'day_mins':day_mins,
                   'day_calls':day_calls,
                   'day_charge':day_charge,
                   'eve_mins':eve_mins,
                   'eve_calls':eve_calls,
                   'eve_charge':eve_charge,
                   'night_mins':night_mins,
                   'night_calls':night_calls,
                   'night_charge':night_charge,
                   'customer_calls':customer_calls
                   }
            features = pd.DataFrame(data,index = [0])
            return features       
        
     
        av = user_input_features()
        av['state'] = av['state'].apply(lambda x: x[:2])         
        av['state'] = av['state'].replace(sc.set_index('State')['Value']) 
        av['voice_plan'] = av['voice_plan'].map({'Yes': "voice_plan_yes" ,'No': "voice_plan_no"})
        av['intl_plan'] = av['intl_plan'].map({'Yes': "intl_plan_yes" ,'No': "intl_plan_no"})
        columns1=['state', 'area_code', 'voice_plan', 'intl_plan']
        for i in columns1:
            x=pd.DataFrame(ohe.fit_transform(av[[i]]), columns=av[i].unique())
            av = pd.concat([x,av], axis=1, join="inner").drop(i,axis=1)
        missing_cols = set(data1) - set(av.columns)
        for col in missing_cols:
            av[col] = 0
        if st.button("Predict"):
            prediction = model.predict(av)
            prediction_proba = model.predict_proba(av)           
            st.subheader('Will Customer Churn ???')
#             st.write('Yes' if prediction[0] == 1 else 'No')
            if prediction[0] == 0:
                st.success('Customer will not Churn')
            elif prediction[0] == 1:
                st.error( 'Customer will Churn')            
            st.subheader('Prediction Probability')
            d77 = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])
            d77 = d77.applymap(lambda x: '{:.2%}'.format(x))
            # Format DataFrame for Streamlit app
            for col, values in d77.iteritems():
                st.write(f'{col}  -  {values[0]}')
            
            
    
    elif option == 'Dataset':
        dataset=st.sidebar.file_uploader("Upload File Here", type = ['csv'])
        if dataset is not None:
            av= pd.read_csv(dataset)
            d=av.copy()
            av['state'] = av['state'].replace(sc.set_index('State')['Value']) 
            av['voice_plan'] = av['voice_plan'].map({'yes': "voice_plan_yes" ,'no': "voice_plan_no"})
            av['intl_plan'] = av['intl_plan'].map({'yes': "intl_plan_yes" ,'no': "intl_plan_no"})   
            columns1=['state', 'area_code', 'voice_plan', 'intl_plan']
            for i in columns1:
                x=pd.DataFrame(ohe.fit_transform(av[[i]]), columns=av[i].unique())
                av = pd.concat([x,av], axis=1, join="inner").drop(i,axis=1)
            if st.button("Predict"):
                predictions = model.predict(av)
                prediction_proba = model.predict_proba(av)
                a = pd.DataFrame(predictions, columns=['Prediction'])
                a['Prediction'] = a['Prediction'].map({1: "yes" ,0: "no"})
                b = pd.DataFrame(prediction_proba, columns=['No-Probability', 'Yes-probability'])
                b = b.applymap(lambda x: '{:.2%}'.format(x))
                c = pd.concat([a, b], axis=1)
                e=pd.concat([d,c], axis=1)
                st.dataframe(e)
                # Add a button to download the data
                def download_button(df):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="my_dataset.csv">Download CSV file</a>'
                    st.markdown(href, unsafe_allow_html=True)
                download_button(e)
                


    