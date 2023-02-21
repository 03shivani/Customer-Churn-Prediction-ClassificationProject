import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
import us
from streamlit_option_menu import option_menu
import association_metrics as am

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
#st.markdown("<h1 style='text-align: center;'>Churn prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; font-family: monospace; color: green;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; font-family: monospace; color: #40e0d0;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; font-family: monospace; color: #7cfc00;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; font-family: monospace; color: #ffdead;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-family: monospace; color: #f5deb3;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; font-family: monospace; color: #ffebcd;'>CHURN PREDICTION</h1>", unsafe_allow_html=True)





#st.markdown("<h1 style='text-align: center; font-family: sans-serif;'>Churn prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center;'>=====================================================================</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; margin-bottom: -40px;'>_____________________________________________________________________</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; margin-top: -40px;'>_____________________________________________________________________</h3>", unsafe_allow_html=True)




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
    df = pd.read_csv("Churn.csv")
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


#changing the data types of both the columns
df['day_charge']=pd.to_numeric(df['day_charge'],errors='coerce')
df['eve_mins']=pd.to_numeric(df['eve_mins'],errors='coerce')

###########################################################################################################################################
 












if selected == 'EDA':
    ## DataSet
#     st.sidebar.markdown("<h3 style='text-align: center;'>Dataset</h3>", unsafe_allow_html=True)
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
#     st.sidebar.markdown("<h2 style='text-align: center;'>----------------------------</h2>", unsafe_allow_html=True)
#     st.sidebar.markdown("<h3 style='text-align: center;'>EDA</h3>", unsafe_allow_html=True)

    EDA = st.sidebar.checkbox('EDA')
    if EDA:
        Report = st.sidebar.selectbox("Select Report",("Descriptive Statistics","Univariate","Bivariate","Multivariate"))
        if Report == "Descriptive Statistics":
            st.markdown("<h3 style='text-align: center;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
            st.table(df.describe())


            st.markdown("<h3 style='text-align: center;'>Correlation</h3>", unsafe_allow_html=True)
            matrix = df.corr()
            matrix = round(matrix,2)
            fig = px.imshow(matrix, width=800, height=800, text_auto=True,color_continuous_scale='Blues_r')
            st.plotly_chart(fig,use_container_width=True)

            st.markdown("<h3 style='text-align: center;'>Categorical Associations</h3>", unsafe_allow_html=True)
            cat = df[['area_code', 'voice_plan', 'intl_plan', 'churn', 'State_names']]
            dff = cat.apply(
                lambda x: x.astype("category") if x.dtype == "O" else x)
            cramers_v = am.CramersV(dff)
            cfit = cramers_v.fit().round(2)
            fig1 = px.imshow(cfit, width=600, height=600, text_auto=True,color_continuous_scale='Blues_r')
            st.plotly_chart(fig1,use_container_width=True)

        if Report == "Univariate":
            selected_col = st.sidebar.selectbox("Select Type",df.columns)
            st.markdown(f"<h2 style='text-align: center;'>Plot for {selected_col}</h2>", unsafe_allow_html=True)
            if selected_col in categorical:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                                    subplot_titles=["Bar Chart", "Pie Chart"], column_widths=[0.6, 0.4])
                fig.add_trace(go.Bar(x=df[selected_col].value_counts().index, y=df[selected_col].value_counts()), row=1, col=1,)
                fig.add_trace(go.Pie(labels=df[selected_col].value_counts().index, values=df[selected_col].value_counts()), row=1, col=2)
                fig.update_layout(height = 600, width = 900, showlegend = False)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig,use_container_width=True)


            else:
                p1 = go.Box(y=df[selected_col], marker=dict(color='Orange'), name=f'{selected_col}')
                p2 = go.Histogram(x=df[selected_col], marker=dict(color='yellowgreen'))
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "Histogram"}]],
                                    subplot_titles=["Box Plot", "Histogram"])

                fig.append_trace(p1, row=1, col=1)
                fig.append_trace(p2, row=1, col=2)
                fig.update_layout(height=500, width=900, showlegend=False)
                # fig.show()
                st.plotly_chart(fig,use_container_width=True)

        if Report == "Bivariate":
            
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
                st.markdown(f"<h2 style='text-align: center;'>{selected_col}</h2>", unsafe_allow_html=True)
                st.plotly_chart(fig,use_container_width=True)

            #                                 ####   Funnel Chart   ####
            else:
                
                pivot_table = df.groupby([selected_col, 'churn']).size().reset_index(name='count')
                pivot_table = pivot_table.pivot(index=selected_col, columns='churn', values='count').fillna(0)
                pivot_table = pivot_table.reset_index().rename_axis(None, axis=1)
                pivot_table.columns = [selected_col, 'No', 'Yes']

                df8 = pivot_table.melt(id_vars=selected_col, value_vars=['No', 'Yes'], value_name='count')
                
                ak= ['#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656', '#1C8356', '#16FF32', '#F7E1A0', '#E2E2E2', '#1CBE4F',
                     '#C4451C', '#DEA0FD', '#FE00FA', '#325A9B', '#FEAF16', '#F8A19F', '#90AD1C', '#F6222E', '#1CFFCE', '#2ED9FF',
                     '#B10DA1', '#C075A6', '#FC1CBF', '#B00068', '#FBE426', '#FA0087', '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D',
                     '#DA16FF', '#222A2A', '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D',
                     '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038','#DB6574']
            
                fig1 = px.funnel(df8, x='variable', y='count'
                                 , color=selected_col
                                        ##  ,color_discrete_sequence=['#DB6574', '#03DAC5','#DB6574', '#03DAC5']
                                 ,color_discrete_sequence=ak
                                
                                 #,color_discrete_sequence = ['red', 'blue', 'orange', 'green']
                                       #   ,title=selected_col
                                )
                fig1.update_traces(textposition='auto', textfont=dict(color='#fff'))
                fig1.update_layout(autosize=True,
                                          margin=dict(t=110, b=50, l=70, r=40),
                                          xaxis_title=' ', yaxis_title=" ",
                                          plot_bgcolor='#0E1117', paper_bgcolor='#0E1117'
                                          #,title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif")
                                          ,font=dict(color='#8a8d93',size=22)
                                         # ,legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                         , height = 600, width = 900)
                st.markdown(f"<h2 style='text-align: center;'>{selected_col} vs Target </h2>", unsafe_allow_html=True)
                st.write("")
                st.markdown("<h3 style='text-align: center;'>Funnel Chart</h3>", unsafe_allow_html=True)
                st.plotly_chart(fig1,use_container_width=True)
                st.write("")
                st.write("")

                   #                                 ####   Tree Map   ####
               
                 
                st.write("")
                st.write("")        
                st.markdown("<h3 style='text-align: center;'>Tree Map</h3>", unsafe_allow_html=True)
                pivot_table = df.groupby([selected_col, 'churn']).size().reset_index(name='count')
                pivot_table = pivot_table.pivot(index=selected_col, columns='churn', values='count').fillna(0)
                pivot_table = pivot_table.reset_index().rename_axis(None, axis=1)
                pivot_table.columns = [selected_col, 'No', 'Yes']

                df7 = pd.melt(pivot_table, id_vars=selected_col, value_vars=['No', 'Yes'])
                fig2 = px.treemap(df7, path=[selected_col, 'variable'], values='value')
                fig2.update_layout(height = 600, width = 900)
                st.plotly_chart(fig2,use_container_width=True)





        if Report == "Multivariate":
            tab1, tab2, tab3 =st.tabs(['Scatter Chart','Map Chart','Sunbrust Chart'])
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
                st.plotly_chart(fig,use_container_width=True)

            with tab2:
                cl1, cl2, cl3, cl4 = st.columns([1, 1, 1, 1])
                selected_col = cl1.selectbox("Select Variable",numeric)
                st.markdown(f"<h2 style='text-align: center;'>Churned customers for each states based on {selected_col}</h2>", unsafe_allow_html=True)
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
                table['%'] = (table['yes']/(table['yes']+table['no']))*100
                table = table.reset_index()
                fig4 = px.choropleth(table, locations='State_codes', color='%', locationmode= "USA-states",
                               color_continuous_scale="PuBu", width=900, height=500)
                fig4.update_geos(fitbounds="locations", visible=False)
                fig4.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'),margin={"r":0,"t":100,"l":0,"b":0},paper_bgcolor='#0E1117',plot_bgcolor='#0E1117')

                st.plotly_chart(fig4,use_container_width=True)

            with tab3:
                df00 = df
                df00['churn'] = np.where(df00['churn']=='yes','churn','non_churn')
                df00['voice_plan'] = np.where(df00['voice_plan'] == 'yes', 'voice_plan_yes', 'voice_plan_no')
                df00['intl_plan'] = np.where(df00['intl_plan'] == 'yes', 'intl_plan_yes', 'intl_plan_no')
#                 grouped = df00.groupby(['area_code', 'voice_plan', 'intl_plan'])['churn'].value_counts().reset_index(
#                     name='count')

#                 # Create the sunburst chart
#                 fig = px.sunburst(grouped, path=['area_code', 'voice_plan', 'intl_plan', 'churn'], values='count',
#                                   color_discrete_sequence=['#416768', '#439EA0'])

#                 fig.update_layout(autosize=True,
#                                   margin=dict(t=110, b=50, l=70, r=40),
#                                   xaxis_title=' ', yaxis_title=" ",
#                                   plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
#                                   title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif"),
#                                   font=dict(color='#8a8d93'),
#                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#                                   )

#                 # Show the plot
#                 st.plotly_chart(fig,use_container_width=True)

                # Create a form to allow the user to select the columns to group by
                categorical = ['area_code', 'voice_plan', 'intl_plan']
                group_by_cols = st.multiselect('Select columns to group by', options=categorical)

                if group_by_cols:
                    # Group the data based on the selected columns
                    grouped = df00.groupby(group_by_cols + ['churn'])['churn'].count().reset_index(name='count')

                    # Create the sunburst chart
                    fig = px.sunburst(grouped, path=group_by_cols + ['churn'], values='count',
                      color_discrete_sequence=px.colors.sequential.Teal)

                    # Customize the chart layout
                    fig.update_layout(autosize=True,
                                      margin=dict(t=110, b=50, l=70, r=40),
                                      xaxis_title=' ', yaxis_title=" ",
                                      plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                      title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif"),
                                      font=dict(color='#8a8d93'),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                      )

                    # Show the plot
                    st.plotly_chart(fig,use_container_width=True)
                else:
                    st.warning('Please select at least one column to group by.')


    









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


data1=['High', 'Low-Medium', 'Low', 'Medium-High', 'intl_plan_no',
       'intl_plan_yes', 'voice_plan_yes', 'voice_plan_no', 'area_code_415',
       'area_code_408', 'area_code_510', 'account_length', 'voice_messages',
       'intl_mins', 'intl_calls', 'intl_charge', 'day_mins', 'day_calls',
       'day_charge', 'eve_mins', 'eve_calls', 'eve_charge', 'night_mins',
       'night_calls', 'night_charge', 'customer_calls']
sc=pd.read_csv("state-category.csv")
unique_states=pd.read_csv("unique_states.csv")
import pickle
model = pickle.load(open('ch.pkl','rb'))
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
                


    
