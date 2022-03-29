import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import gmean
#import matplotlib.pyplot as plt 

import streamlit as st 
import yfinance as yf 

st.title(' Clustering-Based Stock-Index Reduction')

st.write("Adjust the parameters to find if clustering is effective in gaining alpha over the Index (NIFTY50)")

#data=pd.read_csv('C:/Users/sarve/Downloads/neat_nifty2012.csv')
data=pd.read_csv('neat_nifty2012.csv')

x=pd.DataFrame(data.groupby('year_start')['change'].mean()).reset_index()
x.columns=['year_start','average']

data=pd.merge(data,x,on='year_start')
data['alpha']=data['change']/data['average']
data[['Month','year_']]=data['year_start'].astype('str').str.split(expand=True)
data['year_']=data['year_'].astype('float')
data.drop(columns=['year_start.1'],inplace=True)
st.write('Create Ratios')
val="data['Share Capital']"

agree = st.checkbox("I want to create a new ratio",key="1")
if agree==True:

    new_col_name=st.text_input("Enter Name for new Ratio",key="1")
    val="data['Share Capital']"

    val=st.text_input("Enter equation for new Ratio",key="1")
    





    if val:
        data[new_col_name]= eval(val)


agree2 = st.checkbox("I want to create a new ratio",key="2")
if agree2==True:

    new_col_name=st.text_input("Enter Name for new Ratio",key="2")
    val="data['Share Capital']"

    val=st.text_input("Enter equation for new Ratio",key="2")
    






    if val:
        data[new_col_name]= eval(val)


agree3 = st.checkbox("I want to create a new ratio",key="3")
if agree3==True:

    new_col_name=st.text_input("Enter Name for new Ratio",key="3")
    val="data['Share Capital']"

    val=st.text_input("Enter equation for new Ratio",key="3")
    






    if val:
        data[new_col_name]= eval(val)



agree4 = st.checkbox("I want to create a new ratio",key="4")
if agree4==True:

    new_col_name=st.text_input("Enter Name for new Ratio",key="4")
    val="data['Share Capital']"

    val=st.text_input("Enter equation for new Ratio",key="4")
    






    if val:
        data[new_col_name]= eval(val)



agree5 = st.checkbox("I want to create a new ratio",key="5")
if agree5==True:

    new_col_name=st.text_input("Enter Name for new Ratio",key="5")
    val="data['Share Capital']"

    val=st.text_input("Enter equation for new Ratio",key="5")
    






    if val:
        data[new_col_name]= eval(val)


colz=['absolute_profit','alpha_yes']

st.write('CHOOSE YOUR EVALUATION CRITERION TO GO ALONG WITH HIT-RATE/Returns')


option1 = st.selectbox(
   'CRITERIA',
     colz)


if option1=='absolute_profit':
    data['target']=np.where(data['change']>=1,1,0)

elif option1=='alpha_yes':
    data['target']=np.where(data['alpha']>=1,1,0)


st.write(data.head(5))


ll=list(data.columns)
for r in ['year_start','Price_end','year_end','change','average','alpha','Month','year_','target']:
    ll.remove(r)



options = st.multiselect(
     'What are your preferred variables to be included for the Clustering Process?',
     ll)

optionz=options
for v in ['change','target','year_']:
    optionz.append(v)




#st.write('You selected:', options)

min_companies=0
cluster_k=st.number_input('Number of Clusters', min_value=2, max_value=25, value=3)
#cluster_k=5
#nbest=4
nbest=st.number_input('Number of Top Clusters to consider', min_value=1, max_value=cluster_k, value=1)
colz2=['hit_rate','returns']

st.write('CHOOSE YOUR EVALUATION CRITERION')


criterion = st.selectbox(
   'Hit/Returns',
     colz2)
#criterion='hit_rate'
valz=[]
import time




if st.button('Submit'):





    

     

    for k in range(50):
        




        

      values=[]
      


      for i in range(9):

        year='Mar '+str(i+2012)
        yr=i+2012
        sub=data[data['year_']<=yr]
        sub=sub[optionz]


        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(sub.drop(columns=['change','target','year_']))

        kmeans = KMeans(
          init="random",
            n_clusters=cluster_k,
            n_init=10,
            max_iter=300,
            
        )
        
        kmeans.fit(scaled_features)
        sub['cluster']=(kmeans.labels_)

        clusters,numbers,hits,cis=[],[],[],[]
        dat=pd.DataFrame(columns=['cluster','number of companies','hit_rate','returns'])
        for i in range(3):
          su=sub[sub['cluster']==i]
          mn=np.mean(su['target'])
          
          
          ret=np.mean(su['change'])
          le=len(su)
          #print('Cluster '+str(i)+' , number of companies: '+str(le))
          clusters.append(i)
          numbers.append(le)
          #print('')
          #print('target_hit_rate: '+str(mn)+' , returns : '+str(ret))
          cis.append(ret)
          hits.append(mn)
          #print('')
        dat['cluster'],dat['number of companies'],dat['hit_rate'],dat['returns']=clusters,numbers,hits,cis
        dat=dat.sort_values(criterion,ascending=False)
        dat=dat[dat['number of companies']>=min_companies].reset_index(drop=True)
        #print(dat)
        #best_cluster=dat.loc[0,'cluster']
        best_clusters=dat.head(nbest)['cluster'].unique()


        new_sub=data[data['year_']==yr+1]
        new_sub=new_sub[optionz]

        y=new_sub.drop(columns=['change','target','year_'])
        scaler=StandardScaler()
        new_scaled_features = scaler.fit_transform(y)
        identified_clusters = kmeans.predict(new_scaled_features)

        new_sub['cluster']=identified_clusters

        
        nsu=new_sub[new_sub['cluster'].isin(best_clusters)]
        mn=np.mean(nsu['target'])
        ret=np.mean(nsu['change'])
        values.append(ret)
        le=len(nsu)
        #print('Cluster '+str(best_clusters[0])+' , number of companies: '+str(le))
        #print('')
        #print('target_hit_rate: '+str(mn)+' , returns: '+str(ret))
        #print('')
      valz.append((np.prod(values))**(1/8.5))   

    #my_bar = st.progress(0)

    import altair as alt
    if valz:
        st.write(np.mean(valz))

        
        nis=[]
        years=[]

        for year in range(9):




            u=year
            u+=1
            year+=2013
            years.append(year)
            z='-08-31'
            if year==2021:

                z='-03-02'
          

         
            
            dat=yf.download(tickers=['^NSEI'],start=str(year)+'-09-01',end=str(year+1)+z)
          #dat=dat.dropna()
            dat.reset_index(inplace=True)
            nifty=dat.loc[len(dat)-1,'Adj Close']/ dat.loc[0,'Adj Close']
          
            nis.append(nifty)

        dataframe=pd.DataFrame(columns=['years','nis','values'])
        dataframe2=pd.DataFrame(columns=['years','nis','values'])
        dataframe['years'],dataframe['nis'],dataframe['values']=years,np.cumprod(nis),np.cumprod(values)
        dataframe2['years'],dataframe2['nis'],dataframe2['values']=years,np.cumprod(nis),np.cumprod(values)

        x,y=dataframe,dataframe2
        x['val']=x['nis']
        y['val']=y['values']
        x['cat']='nifty'
        y['cat']='clustering'
        dff=pd.concat([x,y])
        dff['years']=dff['years'].astype('int')

        c = alt.Chart(dff).mark_line().encode(
        x='years', y='val',color='cat')
        #st.write(x)

        st.altair_chart(c, use_container_width=True)



