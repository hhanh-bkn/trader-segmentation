if True:
    if True: #import libraries
        import streamlit as st

        import numpy as np 
        import pandas as pd 
        import glob
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        import csv
        import json
        from datetime import datetime  
        from datetime import timedelta 
        from datetime import date
        import time
        import matplotlib.pyplot as plt
        from pandas.plotting import table
        import math
        from scipy import stats
        from scipy.stats import chi2_contingency
        from scipy.stats import chi2
        import warnings
        warnings.filterwarnings('ignore')

        import re
        from time import mktime, strptime

        from sklearn.preprocessing import MinMaxScaler
        from sklearn import metrics 
        from scipy.spatial.distance import cdist 
        from sklearn.cluster import KMeans 
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.cluster import MiniBatchKMeans

        import pickle as pk


        import base64
############################
############################
    # User define functions
    def is_integer_num(n): # Check x is string or is integer
        if isinstance(n, int):
            return True
        if isinstance(n, float):
            return n.is_integer()
        return False

    def is_not_str_num(n): # Check x is not numeric
        if isinstance(n, str):
            return False

    def no_accent_vietnamese(s): # Convert character to normal 
        s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
        s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
        s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
        s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
        s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
        s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
        s = re.sub(r'[ìíịỉĩ]', 'i', s)
        s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
        s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
        s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
        s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
        s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
        s = re.sub(r'[Đ]', 'D', s)
        s = re.sub(r'[đ]', 'd', s)
        return s

    from time import mktime, strptime
    def to_date(user_date):
        if user_date is not None:
            user_date = datetime.strptime(user_date,"%Y-%m-%d")
        else:
            user_date = datetime.now()
        return user_date

    def Present_Outlier_IQR(e,df): # Demonstration IQR Outlier
        df = df.reset_index(drop=True)
        Q1 = np.percentile(df[e],25)
        Q3 = np.percentile(df[e],75)
        IQR = stats.iqr(df[e])
        low = Q1 - 1.5*IQR
        high = Q3 + 1.5*IQR
        outliers = df[e][(df[e]<low) | (df[e]>high)]
        percent = outliers.size/(df[e].shape[0])
        print('*'*40)
        return percent,low,high
############################
############################
    def xu_ly_du_lieu(df_pre):
        df_pre = df_pre.drop_duplicates(subset=['user_id']).reset_index(drop=True)
        df_pre['real_user'] = (df_pre['city'].isna() == False) & (df_pre['volume'].isna() == False)

        st.write('Quantity by registration: ', df_pre['real_user'].value_counts()[0],'Email legit: ',df_pre['real_user'].value_counts()[1])
        figure = plt.subplots(figsize=(8,6))
        df_pre['real_user'].value_counts().plot(kind='bar')
        plt.title('User confirmed email & Spam email')
        st.pyplot(figure[0])

        df_pre = df_pre[df_pre['real_user'] == True]
        shape_before = df_pre.shape[0]
        df_pre = df_pre.dropna().reset_index(drop=True)
        shape_after = df_pre.shape[0]
        st.write('Users Verifired Kyc & Volume more than 0')
        st.write('Spam email: ',shape_before, 'Email confirmed:', shape_after)
        st.write('*'*40)

        # Remove outliers `age`
        percent,low,high = Present_Outlier_IQR('age',df_pre)
        print(100*percent, high)

        # Remove 3.45% Outliers
        shape_before = df_pre.shape[0]
        df_pre = df_pre[(df_pre['age'] >= 14) & (df_pre['age'] <= 50)].reset_index(drop=True)
        shape_after = df_pre.shape[0]
        st.write('Quantity by registration: ',shape_before,'Real users: ',shape_after)
        # st.write('Removed 3.45% Outliers `age` variable')

        df_pre['vung'] = df_pre['Vung'].map({'I':0,'II':1,'III':2,'IV':3 })
        df_filtered_users = df_pre[(df_pre['qty_device'] >= 18) | (df_pre['qty_swap_bank'] >= 4)].reset_index(drop=True)
        df_filtered_users = df_filtered_users.sort_values('qty_device', ascending=False).reset_index(drop=True)
        df_final = df_pre[~((df_pre['qty_device'] >= 18) | (df_pre['qty_swap_bank'] >= 4))].reset_index(drop=True)


        return df_final, df_filtered_users

    def chuan_doan_du_lieu(df_final):
        features = ['volume','created_days', 'age', 'qty_swap_login', 'vung','qty_device']
        scaler = MinMaxScaler()
        scaler = scaler.fit(df_final[features])
        X = scaler.transform(df_final[features])
        X = pd.DataFrame(X, columns=df_final[features].columns)
        X['volume'] = X['volume'] * 100
        
        N_clusters = 3
        kmeans_labels=KMeans(n_clusters=N_clusters).fit_predict(X)
        kmean_al = KMeans(n_clusters=N_clusters).fit(X)
        kmeans_score = metrics.silhouette_score(X,kmeans_labels, metric='euclidean')
        k_means_dv_score = metrics.davies_bouldin_score(X,kmeans_labels) # Davies Bouldin Score
        st.dataframe(pd.DataFrame({'Algorithms':'KMeans','Silhouette Score':kmeans_score, 'Davies Bouldin Score':k_means_dv_score},index=[0]))

        kmeanModel = KMeans(n_clusters=N_clusters)
        kmeanModel.fit(X)   

        centroids = kmeanModel.cluster_centers_
        labels = kmeanModel.labels_
        df_final['label'] = labels
        centroids[:,0] = centroids[:,0]/100
        centers_df = pd.DataFrame(centroids,columns=X.columns)
        minmax_centroids = scaler.inverse_transform(centroids)
        centers_df_minmax_centroids = pd.DataFrame(minmax_centroids,columns=X.columns)
        st.write('*'*40)
        st.write('Center clusters')
        st.dataframe(centers_df_minmax_centroids)

        st.write('*'*40)
        st.dataframe(df_final['label'].value_counts())
        figure = plt.subplots(figsize=(8,6))
        df_final['label'].value_counts().plot(kind='bar')
        plt.title('Quantity of cluster')
        plt.yscale("log")
        st.pyplot(figure[0])
        return df_final,centers_df_minmax_centroids,scaler,kmeanModel

    def ve_bieu_do(df_final,centers_df_minmax_centroids):
        figure = plt.subplots(figsize=(12,12))
        plt.subplot(221)
        plt.scatter(df_final.age, df_final.volume, c=df_final.label)
        plt.scatter(centers_df_minmax_centroids['age'], centers_df_minmax_centroids['volume'], marker='X',c='red')
        plt.xlabel('age')
        plt.ylabel('volume')

        plt.subplot(222)
        plt.scatter(df_final.created_days, df_final.volume, c=df_final.label)
        plt.scatter(centers_df_minmax_centroids['created_days'], centers_df_minmax_centroids['volume'], marker='x',c='red')
        plt.xlabel('created_days')
        plt.ylabel('volume')

        plt.subplot(223)
        plt.scatter(df_final.qty_swap_login, df_final.volume, c=df_final.label)
        plt.scatter(centers_df_minmax_centroids['qty_swap_login'], centers_df_minmax_centroids['volume'], marker='X',c='red')
        plt.xlabel('qty_swap_login')
        plt.ylabel('volume')
        plt.xlim([0.0, 500.0])

        plt.subplot(224)
        plt.scatter(df_final.qty_device, df_final.volume, c=df_final.label)
        plt.scatter(centers_df_minmax_centroids['qty_device'], centers_df_minmax_centroids['volume'], marker='x',c='red')
        plt.xlabel('qty_device')
        plt.ylabel('volume')
        
        plt.suptitle('Scatter plot filterd by volume & features',fontsize=20)
        st.pyplot(figure[0])

        st.markdown("""
*Bài toán được chia thành 2 nhóm lớn -> và 1 nhóm lớn gồm 3 nhóm nhỏ <br/>
(các con số chia nhóm cũng thay đổi theo thời gian do volume các users càng ngày càng tăng)
<br/>
- Nhóm 1 (thiết bị >=18 hoặc dùng trên 4 tài khoản ngân hàng) đây là những users đặc biệt cần lưu ý để check scam, rửa tiền, nghiên cứu thuật toán của họ.
<br/>
- Nhóm 2.1 users giao dịch với khối lượng lớn (dưới 17 tỷ)
- Nhóm 2.2 users giao dịch với khối lượng lớn (trên 17 tỷ dưới 80 tỷ)
- Nhóm 2.3 users giao dịch với khối lượng lớn (trên 80 tỷ) độ tuổi từ 30-40, tạo tài khoản từ 200 ngày trở lên, số lần logout-login vừa phải không quá nhiều, sử dụng ~3 thiết bị.""", unsafe_allow_html=True,)
        return
############################
############################
    def name():
        name = st.text_input('Input your name:')
        name = name.capitalize()
        st.write('Labeling trader: '+name)
        return name
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    def volume():
        volume=st.text_input('- *Volume (10^9 _ Billion):')
        Volume=0
        if volume!='':
            if is_number(volume):
                if float(volume)>0:
                    volume = float(volume)
                    st.write('Total volume transaction is: ','{:,.0f}'.format(volume*(10**9)))
                    st.success('Ok!')
                    Volume=int(volume*(10**9))
                else:
                    st.warning('Volume must >0')
            else:
                st.warning('Only input number')
        else:
            st.write('Only input number')
        return Volume

    def tuoi():
        age = st.slider('How old are you:', min_value=14, max_value=50)
        return age

    def bank():
        Bank = st.slider('Quantity of baning account:', min_value=0, max_value=20)
        return Bank

    def device():
        qty_device=st.text_input('- *Input quantity device:')
        Qty_device=0
        if qty_device!='':
            if qty_device.isdigit():
                if int(qty_device)>0:
                    st.success('Ok!')
                    Qty_device=int(qty_device)
                else:
                    st.warning('')
            else:
                st.warning('Only input number')
        else:
            st.write('Only input number')
        return Qty_device

    def created_days():
        days=st.text_input('- *Input created_days:')
        Days=0
        if days!='':
            if days.isdigit():
                if int(days)>0:
                    st.success('Ok!')
                    Days=int(days)
                else:
                    st.warning('Only input number')
            else:
                st.warning('Only input number')
        else:
            st.write('Only input number')
        return Days

    def login():
        qty_login=st.text_input('- *Input quantity swap login:')
        Qty_login=0
        if qty_login!='':
            if qty_login.isdigit():
                if int(qty_login)>0:
                    st.success('Ok!')
                    Qty_login=int(qty_login)
                else:
                    st.warning('')
            else:
                st.warning('Only input number')
        else:
            st.write('Only input number')
        return Qty_login

    def vungkinhte():
        st.write('Choosing Economic Zone:')
        Vung = st.selectbox('Choosing Economic Zone: ',['I','II','III','IV'])
        if Vung == 'I': vung=0
        elif Vung == 'II': vung=1
        elif Vung == 'III': vung=2
        elif Vung == 'IV': vung=3
        st.success('Ok!')
        return vung
    
    def nhap_thong_tin():
        Name  = name()
        Age = tuoi()
        Volume = volume()
        Device = device()
        Created_days = created_days()
        Login = login()
        Vungkinhte = vungkinhte()
        Bank = bank()
        return Name, Age, Volume, Device, Created_days,Login,Vungkinhte,Bank
############################
    def all_to_model(df_pre):
        df_pre = df_pre.drop_duplicates(subset=['user_id']).reset_index(drop=True)
        df_pre['real_user'] = (df_pre['city'].isna() == False) & (df_pre['volume'].isna() == False)

        df_pre = df_pre[df_pre['real_user'] == True]
        df_pre = df_pre.dropna().reset_index(drop=True)
        df_pre = df_pre[(df_pre['age'] >= 14) & (df_pre['age'] <= 50)].reset_index(drop=True)

        df_pre['vung'] = df_pre['Vung'].map({'I':0,'II':1,'III':2,'IV':3 })
        df_filtered_users = df_pre[(df_pre['qty_device'] >= 18) | (df_pre['qty_swap_bank'] >= 4)].reset_index(drop=True)
        df_filtered_users = df_filtered_users.sort_values('qty_device', ascending=False).reset_index(drop=True)
        df_final = df_pre[~((df_pre['qty_device'] >= 18) | (df_pre['qty_swap_bank'] >= 4))].reset_index(drop=True)

        features = ['volume','created_days', 'age', 'qty_swap_login', 'vung','qty_device']
        scaler = MinMaxScaler()
        scaler = scaler.fit(df_final[features])
        X = scaler.transform(df_final[features])
        X = pd.DataFrame(X, columns=df_final[features].columns)
        X['volume'] = X['volume'] * 100
        
        N_clusters = 3
        kmeanModel = KMeans(n_clusters=N_clusters)
        kmeanModel.fit(X)   

        centroids = kmeanModel.cluster_centers_
        labels = kmeanModel.labels_
        df_final['label'] = labels
        centroids[:,0] = centroids[:,0]/100
        minmax_centroids = scaler.inverse_transform(centroids)
        centers_df_minmax_centroids = pd.DataFrame(minmax_centroids,columns=X.columns)
        
        return df_final,centers_df_minmax_centroids,scaler,kmeanModel
############################
    def thong_bao(df_final,centers_df_minmax_centroids,scaler,kmeanModel, Name, Age, Volume, Device, Created_days,Login,Vungkinhte,Bank):
        features = ['volume', 'created_days', 'age', 'qty_swap_login', 'vung', 'qty_device']
        list_feature = [[Volume, Created_days, Age,  Login, Vungkinhte, Device]]
        A = pd.DataFrame(list_feature, columns=features)
        B = scaler.transform(A)
        B = pd.DataFrame(B, columns=features)
        B['volume'] = B['volume']*100

        # st.dataframe(A)

        if ((Device < 18) and (Bank < 4)):
            # st.write(kmeanModel.predict(B))
            if Volume < 9e9: result = 'Beginer Trader' 
            elif Volume < 60e9 : result = 'Normal Trader' 
            elif Volume > 60e9 : result = 'Vip Trader' 
            else: result = 'Terminal Trader'
            st.write('User ',Name,' is a',result)
            figure = plt.subplots(figsize=(10,8))
            scatter = plt.scatter(df_final.age, df_final.volume, c=df_final.label,cmap='viridis')
            plt.scatter(centers_df_minmax_centroids['age'], centers_df_minmax_centroids['volume'], marker='X',c='red')
            plt.scatter(A['age'], A['volume'], marker='*',c='olive',s=1000)
            plt.xlabel('age')
            plt.ylabel('volume')
            plt.legend(*scatter.legend_elements())
            st.pyplot(figure[0])
        elif ((Device >= 18) and (Bank >=4)):
            st.write('User ',Name, 'were swapped bank ', Bank,'time & swapped device ',Device,' -> Higher probability Scame, please check info this user in the Internet')
        elif Bank >=4:
            st.write('User ',Name, 'was swapped bank ', Bank,' -> Higher probability Scame, please check info this user in the Internet')
        else:
            st.write('User ',Name, 'was swapped device ', Device,' -> Higher probability Scame, please check info this user in the Internet')
        return 


    def download_csv(df):
        """
        Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href =f'Ok! Press<a href="data:file/csv;base64,{b64}"\
            download="labeled_.csv"   > vào đây </a>to dowload file'
        st.markdown(href, unsafe_allow_html=True)
        return
    if __name__=='__main__':
        main()