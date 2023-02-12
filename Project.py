import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


import plotly.graph_objects as go
from plotly.subplots import make_subplots


import plotly.express as px


from sklearn.linear_model import LinearRegression



#Set Page Layout
st.set_page_config(layout="wide")


# title
st.markdown("<h1 style='text-align: center;'>Classification</h1>", unsafe_allow_html=True)
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

import us
states = us.states.mapping("abbr","name")
State = df['state'].map(lambda x: states.get(x, x))

df.insert(loc=0, column='State_names', value=State)
df.rename(columns = {'state':'State_codes'}, inplace = True)

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
    Report = st.sidebar.selectbox("Select Report",("Descriptive Statistics","Univeriate","Baiveriate","Multivariate", "My", "Pritesh"))
    if Report == "Descriptive Statistics":
        st.markdown("<h3 style='text-align: center;'>Describe</h3>", unsafe_allow_html=True)
        st.table(df.describe())
        st.markdown("<h3 style='text-align: center;'>Correlation</h3>", unsafe_allow_html=True)
        matrix = df.corr()
        matrix = round(matrix,2)
        fig = px.imshow(matrix, width=800, height=800, text_auto=True)
        st.plotly_chart(fig)

    if Report == "Univeriate":
        selected_col = st.sidebar.selectbox("Select Type",('categorical','numeric'))
        if selected_col == "categorical":
            st.write("categorical")
            for i in categorical:
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                                    subplot_titles=["Bar Chart", "Pie Chart"], column_widths=[0.6, 0.4])
                fig.add_trace(go.Bar(x=df[i].value_counts().index, y=df[i].value_counts()), row=1, col=1)
                fig.add_trace(go.Pie(labels=df[i].value_counts().index, values=df[i].value_counts()), row=1, col=2)
                fig.update_layout(
                    {'title': {'text': f"plots for {i}", 'x': 0.5, 'y': 0.9, 'font_size': 25, 'font_color': 'Blue'}},
                height = 600, width = 900, showlegend = False)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig)
        else:
            st.write("numeric")
            for i in numeric:
                p1 = go.Box(y=df[i], marker=dict(color='Orange'), name=f'{i}')
                p2 = go.Histogram(x=df[i], marker=dict(color='yellowgreen'))

                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "Histogram"}]],
                                    subplot_titles=["Box Plot", "Histogram"])

                fig.append_trace(p1, row=1, col=1)
                fig.append_trace(p2, row=1, col=2)
                fig.update_layout(
                    {'title': {'text': f"plots for {i}", 'x': 0.5, 'y': 0.9, 'font_size': 25, 'font_color': 'Blue'}}
                    , height=600, width=900, showlegend=False)
                #fig.show()
                st.plotly_chart(fig)

    if Report == "Baiveriate":
        selected_col = st.sidebar.selectbox("Select Column", df.columns)
        cl1, cl2 = st.columns([1,1])
        colo = ['#012D9C', '#7097FF']

        #df1 = df.groupby(by=[selected_col,'churn']).size().reset_index(name="counts")
        #fig = px.bar(df1, x=selected_col, y='counts', color="churn", color_continuous_scale=colo,title="Count plot", width=900, height=500)
        #fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
        #st.plotly_chart(fig)
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
                                  plot_bgcolor='#2d3035', paper_bgcolor='#2d3035',
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #fig4 = px.choropleth(df2, locations='state', color='counts', locationmode= "USA-states",
        #                   color_continuous_scale="PuBu", width=900, height=500)
        #fig4.update_geos(fitbounds="locations", visible=False)
        #fig4.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'),margin={"r":0,"t":100,"l":0,"b":0},paper_bgcolor='#000000',plot_bgcolor='#000000')
        #st.plotly_chart(fig4)
        #df1 = df.groupby(by=['area.code']).size().reset_index(name="counts")
        #fig2 = px.pie(df1, values='counts', names='area.code', width=400, height=400)
        #st.plotly_chart(fig2)
        #st.write("Hello")

    if Report == "My":

        x_axis = st.sidebar.selectbox("Select X_axis", numeric)
        Y_axis = st.sidebar.selectbox("Select Y_axis", numeric)
        co1, co2, co3, co4 = st.columns([1,1,1,1])
        uni2 = df['area_code'].unique()
        uni3 = df['voice_plan'].unique()
        uni4 = df['intl_plan'].unique()
        f2 = co2.selectbox("Area_code", uni2)
        f3 = co3.selectbox("Voice_plan", uni3)
        f4 = co4.selectbox("intl_plan", uni4)
        #df = df[df['State_names'] == f1]
        df = df[df['area_code'] == f2]
        df = df[df['voice_plan'] == f3]
        df = df[df['intl_plan'] == f4]
        fig = px.scatter(df, x=x_axis, y=Y_axis,
                         color="churn",width=900,height=500)#hover_name="country", log_x=True,
        st.write("Buble chart")
        st.plotly_chart(fig)

    if Report == "Pritesh":
        # Group the data by 'area_code' and 'churn' and compute the count
        pivot_table = df.groupby(['area_code', 'churn']).size().reset_index(name='count')

        # Pivot the data to create a count table
        pivot_table = pivot_table.pivot(index='area_code', columns='churn', values='count').fillna(0)

        # # Reset the index and rename the columns
        pivot_table = pivot_table.reset_index().rename_axis(None, axis=1)
        pivot_table.columns = ['area_code', 'No', 'Yes']

        # Convert the pivot table into a format suitable for Plotly
        df7 = pd.melt(pivot_table, id_vars='area_code', value_vars=['No', 'Yes'])

        # Plot the data using Plotly express
        fig = px.treemap(df7, path=['area_code', 'variable'], values='value')

        # Show the plot
        st.plotly_chart(fig)

        # Convert the pivot table into a format suitable for Plotly
        df8 = pivot_table.melt(id_vars='area_code', value_vars=['No', 'Yes'], value_name='count')

        # Plot the data using Plotly express
        fig = px.funnel(df8, x='variable', y='count', color='area_code'
                        , color_discrete_sequence=['#DB6574', '#03DAC5']
                        , title='area_code')

        fig.update_traces(textposition='auto', textfont=dict(color='#fff'))
        fig.update_layout(autosize=True,
                          margin=dict(t=110, b=50, l=70, r=40),
                          xaxis_title=' ', yaxis_title=" ",
                          plot_bgcolor='#2d3035', paper_bgcolor='#2d3035',
                          title_font=dict(size=25, color='#a5a7ab', family="Muli, sans-serif"),
                          font=dict(color='#8a8d93', size=22),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                          )
        # Show the plot
        st.plotly_chart(fig)

        # Group the data by 'area_code' and 'State_names'
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



st.sidebar.markdown("<h2 style='text-align: center;'>----------------------------</h2>", unsafe_allow_html=True)




## Data Pre-Processing
#Data = df
#columns = ['State_names','area_code','voice_plan','intl_plan']
#label_encoder = LabelEncoder()
#for col in columns:
#    Data[col] = label_encoder.fit_transform(Data[col])
#
#Data['churn'] = np.where(Data['churn']=="yes",1,0)



## train test split part
#X = Data.drop(['churn'],axis=1)
#y = Data['churn']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#shapee = st.sidebar.checkbox("Shape of train and test data",)
#if shapee:
#     st.write('shape of x train data',X_train.shape)
#     st.write('shape of x test data',X_test.shape)
#     st.write('shape of y test data',y_test.shape)
#     st.write("shape of y train data",y_train.shape)
#
#
#g=GaussianNB()
#b=BernoulliNB()
#KN=KNeighborsClassifier()
##SVC= SVC()
#D=DecisionTreeClassifier()
#R=RandomForestClassifier()
#Log=LogisticRegression()
#XGB=XGBClassifier()
#
#st.sidebar.markdown("<h3 style='text-align: center;'>Model Manager</h3>", unsafe_allow_html=True)
##st.sidebar.subheader("Model Manager")
#item = st.sidebar.selectbox("Select Method",('GaussianNB', 'BernoulliNB', 'KNeighborsClassifier', 'DecisionTreeClassifier',
#                  'RandomForestClassifier', 'LogisticRegression', 'XGBClassifier'))
#
#if item == "GaussianNB":
#    item1 = g
#elif item == "BernoulliNB":
#    item1 = b
#elif item == "KNeighborsClassifier":
#    item1 = KN
#elif item == "DecisionTreeClassifier":
#    item1 = D
#elif item == "RandomForestClassifier":
#    item1 = R
#elif item == "LogisticRegression":
#    item1 = Log
#else:
#    item1 = XGB
#
#
#item1.fit(X_train, y_train)
#item1.predict(X_test)
#a = accuracy_score(y_test, item1.predict(X_test))
#b = precision_score(y_test, item1.predict(X_test))
#c = recall_score(y_test, item1.predict(X_test))
#d = f1_score(y_test, item1.predict(X_test))
#
#abc = {'name': ["accuracy_score", "precision_score", "recall_score", "f1_score"], 'Result': [a, b, c, d]}
#
#Result = st.sidebar.checkbox("Result")
#if Result:
#    st.table(abc)
