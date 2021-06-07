if True:
    if True: #import libraries
        import streamlit as st

        import numpy as np
        import pandas as pd
        
        import pickle as pk
        from PIL import Image

        import functions as ft

    ############################
    def main():
        st.title('Demo Customer Segmentation')  
        # with open('model.pkl','rb') as f:
        #     model=pk.load(f)
        menu=['Modeling','Customer labeling','Clustering & Save file']
        choice=st.sidebar.selectbox('Menu',menu)
        # st.subheader(choice)

        if choice=='Modeling':
            #Load dữ liệu
            image = Image.open('image/avatar.jpg')
            st.image(image, caption=None,use_column_width=True,width=None)
            st.subheader('1. Upload data')

            file_csv = st.file_uploader("Upload file csv", type=([".csv"]))
            if st.button('Upload data!'):
                file_csv='temp_data.csv'
            if file_csv:
                data=pd.read_csv(file_csv)
                if 'user_id' not in data.columns:
                    st.write('Invalid data, please choose another data !')
                else:
                    #Xử lý dữ liệu
                    st.subheader('2. Data preprocessing')
                    df_final, df_filtered_users = ft.xu_ly_du_lieu(data)

                    st.write('1. Data normal users')
                    st.write('-',df_final.shape[0],' users')
                    st.dataframe(df_final.head())
                    st.write('*'*40)
                    st.write('2. Data special users (qty_device>=18 & qty_bank>=4)')
                    st.write('-',df_filtered_users.shape[0],' records')
                    st.dataframe(df_filtered_users[['user_id','volume', 'qty_device','qty_swap_bank', 'created_days', 'age', 'qty_swap_login', 'vung']].head())

                    #Chuẩn đoán dữ liệu
                    st.subheader('3. Modelling')
                    df_final,centers_df_minmax_centroids,scaler,kmeanModel = ft.chuan_doan_du_lieu(df_final)

                    #Vẽ biểu đồ
                    st.subheader('4. Model deployment and demonstration')
                    ft.ve_bieu_do(df_final,centers_df_minmax_centroids)
                    image = Image.open('image/tree.png')
                    st.image(image, caption=None,use_column_width=True,width=None)

        elif choice=='Customer labeling':
            file_csv='temp_data.csv'
            df_pre=pd.read_csv(file_csv)
            Name, Age, Volume, Device, Created_days,Login,Vungkinhte, Bank = ft.nhap_thong_tin()
            df_final,centers_df_minmax_centroids,scaler,kmeanModel = ft.all_to_model(df_pre)


            if Volume > 0:
                if st.button('Prediction'):
                    ft.thong_bao(df_final,centers_df_minmax_centroids,scaler,kmeanModel, Name, Age, Volume, Device, Created_days,Login,Vungkinhte,Bank)
                else:
                    st.write('Click to `Prediction` to clustering')
            else:
                st.write('*'*40)
                st.write('Hmm. Please input features to define segments customer !')
        else:
            #Load dữ liệu
            st.subheader('Clustering & Save file')

            file_csv = st.file_uploader("Upload file csv tại đây", type=([".csv"]))
            if st.button('Upload data!'):
                file_csv='temp_data.csv'
            if file_csv:
                data=pd.read_csv(file_csv)
                if 'user_id' not in data.columns:
                    st.write('Invalid data, please choose another data !')
                else:
                    #Xử lý dữ liệu
                    st.subheader('2. Data preprocessing')
                    df_final, df_filtered_users = ft.xu_ly_du_lieu(data)

                    st.write('1. Data normal users')
                    st.dataframe(df_final.head())
                    st.write('*'*40)
                    st.write('2. Data special users (qty_device>=18 & qty_bank>=4)')
                    st.dataframe(df_filtered_users[['user_id','volume', 'qty_device','qty_swap_bank', 'created_days', 'age', 'qty_swap_login', 'vung']].head())

                    #Chuẩn đoán dữ liệu
                    st.subheader('3. Clustering')
                    df_final,centers_df_minmax_centroids,scaler,kmeanModel = ft.chuan_doan_du_lieu(df_final)
                    
                    #Tải dữ liệu xuống
                    st.write('*'*40)
                    ft.download_csv(df_final[['user_id','label']])


            else:
                image = Image.open('image/giaodiensan.png')
                st.image(image, caption=None,use_column_width=True,width=None)
        return


    if __name__=='__main__':
        main()





































































