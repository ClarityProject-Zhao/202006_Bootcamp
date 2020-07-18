import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt

path = os.path.abspath(os.getcwd())
data_file = os.path.join(path, 'time_series_covid19_confirmed_US.csv')
data = pd.read_csv(data_file, sep=',', index_col=0)

#
data_byState=data.loc[:,['Province_State']+data.columns[10:].tolist()].groupby('Province_State').sum()

# matplotlib line plot
cols=data_byState.columns.tolist()

plt.close('all')
plt.figure(figsize=[10,6])
state='New York'
plt.plot(data_byState.loc[state,:])
x_ticks_mask=np.arange(0,len(data_byState.columns))
x_ticks= [x if (cols.index(x) in x_ticks_mask[::20]) else "" for x in cols]
plt.xticks(ticks=x_ticks)
plt.title(f'Total Confirmed Cases: {state}')

#seaborn line plot
plt.close('all')
plt.figure(figsize=[10,6])
state='New York'
df=pd.DataFrame(data_byState.loc[state,:])
df=df.reset_index()
df['index']=df['index'].apply(lambda x: dt.datetime.strptime(x,"%m/%d/%y"))
sns.lineplot(data=df,x='index',y=state)


#seaborn other plots
state=["New York","Florida"]
df=pd.DataFrame(data_byState.loc[state,:]).transpose()
df=df.reset_index()
df['index']=df['index'].apply(lambda x: dt.datetime.strptime(x,"%m/%d/%y"))
sns.relplot(data=df,x=state[0],y=state[1])

top_states=data_byState.sort_values('7/16/20',ascending=False).index[0:9].tolist()
sns.catplot(data=data[data['Province_State'].isin(top_states)],x='Province_State',y='7/16/20')

#plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"

# line plot
df=data_byState.transpose()
df=df.reset_index()
df['index']=df['index'].apply(lambda x: dt.datetime.strptime(x,"%m/%d/%y"))
fig = px.line(df, x="index", y="New York")
fig.show()

# trace race
trace1 = go.Scatter(x=df['index'],y=df['New York'],mode='lines',name="New York")
trace2= go.Scatter(x=df['index'],y=df['California'],mode='lines',name="California")
trace3 = go.Scatter(x=df['index'],y=df['Florida'],mode='lines',name="Florida")
trace4 = go.Scatter(x=df['index'],y=df['Texas'],mode='lines',name="Texas")


frames = [dict(data= [dict(type='scatter',
                           x=df['index'][:k+1],
                           y=df['New York'][:k+1]),
                      dict(type='scatter',
                           x=df['index'][:k + 1],
                           y=df['California'][:k + 1]),
                      dict(type='scatter',
                           x=df['index'][:k + 1],
                           y=df['Florida'][:k + 1]),
                      dict(type='scatter',
                           x=df['index'][:k + 1],
                           y=df['Texas'][:k + 1]),
                     ],
               traces= [0, 1, 2, 3],
              )for k  in  range(0, len(df)-1)]
layout=go.Layout(
        xaxis=dict(range=['2020-1-23', '2020-7-16'], autorange=False),
        yaxis=dict(range=[0, 500000], autorange=False),
        title="Coronavirus Trend in Major US States",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Show Trend",
                          method="animate",
                          args=[None,dict(frame=dict(duration=3,
                                                  redraw=False),
                                                  transition=dict(duration=0.2),
                                                  fromcurrent=True,
                                                  mode='immediate')])])])


fig = go.Figure(data=[trace1, trace2, trace3, trace4], frames=frames,layout=layout)
fig.show()



##play with seaborn
#iris data
iris = sns.load_dataset("iris")
sns.jointplot(x="sepal_length", y="petal_length", data=iris);

#cloting rating data from week3
path = os.path.abspath(os.getcwd())
data_file = os.path.join(path, 'rating_data.csv')
data_rating = pd.read_csv(data_file, sep=',', index_col=0)

sns.catplot(data=data_rating,x='Rating',y='Age')
sns.boxplot(data=data_rating,y='Rating',x='Department Name')
sns.pairplot(data_rating[['Rating','Age','Department Name']],hue='Department Name')