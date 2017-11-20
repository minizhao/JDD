import pandas as pd
import numpy as np
import pickle as pkl
import sys

class Data_loader():
    # 建一个类用来加载某个店铺的某个月份统计信息
    def __init__(self,df_comment,df_ads,df_order,df_product,df_sales_sum,df_Sales_Forecast):

        self.df_comment = df_comment
        self.df_ads = df_ads
        self.df_order = df_order
        self.df_product = df_product
        self.df_sales_sum = df_sales_sum
        self.df_Sales_Forecast = df_Sales_Forecast

    def conver_comment(self, df,last_m_comment):
        # df = pd.read_csv(data_file)
        # 条件查询，得到这个商铺和这个月的信息
        df = df[np.logical_and(df.shop_id == self.shop_id, df.create_dt.str.contains(self.month))]
        if last_m_comment=='':last_m_comment=np.zeros(9)
        # print(df)
        # 返回统计数9
        f=[np.sum(df.bad_num), np.sum(df.cmmt_num), \
              np.sum(df.dis_num), np.sum(df.good_num), np.sum(df.mid_num),\
              np.sum(df.bad_num)/(np.sum(df.cmmt_num)+1),np.sum(df.dis_num)/(np.sum(df.cmmt_num)+1),\
              np.sum(df.good_num)/(np.sum(df.cmmt_num)+1),np.sum(df.mid_num)/(np.sum(df.cmmt_num)+1)]
        f.extend(np.array(f)-np.array(last_m_comment))
        return f
    def conver_ads(self,df,last_m_ads):
        # df = pd.read_csv(data_file)
        df = df[np.logical_and(df.shop_id == self.shop_id,df.create_dt.str.contains(self.month))]
        if last_m_ads=='':last_m_ads=0#只有一个消费数
        if df.shape[0]==0:
            return [0,0,0]
        f=[np.sum(df.charge),np.sum(df.consume)]
        f.append(f[1]-last_m_ads)
        return f
    def conver_order(self,df):
        # df = pd.read_csv(data_file)
        df = df[np.logical_and(df.shop_id == self.shop_id,df.ord_dt.str.contains(self.month))]
        # print(df)
        return [np.sum(df.sale_amt),np.sum(df.offer_amt),np.sum(df.offer_cnt),\
              np.sum(df.rtn_cnt),np.sum(df.rtn_amt),np.sum(df.ord_cnt),len(df.pid.drop_duplicates()),\
              np.sum(df.user_cnt), np.sum(df.rtn_cnt)/(np.sum(df.ord_cnt)+1),np.sum(df.sale_amt)/(np.sum(df.ord_cnt)+1)]
    #
    def conver_product(self,df):
        # df = pd.read_csv(data_file)
        df = df[df.shop_id == self.shop_id]
        # return (df)
        return [len(df.brand.drop_duplicates()),len(df.cate.drop_duplicates()),len(df.pid.drop_duplicates())]

    def create_train_data(self):

        feature_list=[0,0,0]
        label_list=[0,0,0]
        all_shop=self.df_ads.shop_id.drop_duplicates()
        months=list(set([x[:7] for x in self.df_ads.create_dt.drop_duplicates()]))#提取月份
        months.sort()
        for shop_id in all_shop:
            self.shop_id=shop_id
            last_m_comment=''
            last_m_ads=''

            for month in months:
                self.month=month
                f_comment=dl.conver_comment(df_comment,last_m_comment)
                f_ads=dl.conver_ads(df_ads,last_m_ads)

                f_order=dl.conver_order(df_order)
                f_product=dl.conver_product(df_product)

                feature = np.concatenate((f_comment, f_ads,f_order,f_product,[shop_id]))
                label = df_sales_sum[np.logical_and(df_sales_sum.shop_id==shop_id,df_sales_sum.dt.str.contains(month))]
                
                if f_comment!='': last_m_comment=f_comment[:9]
                if f_ads!='': last_m_ads=f_ads[1]
                
                if label.shape[0]!=0:
                    feature_list.append(feature)
                    label_list.append(label.sale_amt_3m.tolist()[0])
                    if len(label_list)%100==0:
                        print(str(len(label_list))+'/'+str(len(months)*len(all_shop)))
        assert len(feature_list)==len(label_list)
        pkl.dump(feature_list,open('save/feature_list.pkl','wb'))
        pkl.dump(label_list,open('save/label_list.pkl','wb'))
        


    def create_test_data(self):
        forecast_feature_list=[]
        forecast_shop_list=[]
        all_forecast_shop=self.df_Sales_Forecast.shop_id.drop_duplicates()
        for shop_id in all_forecast_shop:
            self.shop_id=shop_id
            last_m_comment=''
            last_m_ads=''
            self.month='2017-04'

            f_comment=dl.conver_comment(df_comment,last_m_comment)
            f_ads=dl.conver_ads(df_ads,last_m_ads)

            f_order=dl.conver_order(df_order)
            f_product=dl.conver_product(df_product)

            feature = np.concatenate((f_comment,f_ads,f_order,f_product,[shop_id]))
            if f_comment!='': last_m_comment=f_comment[:9]
            if f_ads!='': last_m_ads=f_ads[1]
                
            forecast_feature_list.append(feature)
            forecast_shop_list.append(shop_id)

            if len(forecast_feature_list)%100==0:
                print(str(len(forecast_feature_list))+'/'+str(len(all_forecast_shop)))

        pkl.dump(forecast_feature_list,open('save/forecast_feature_list.pkl','wb'))
        pkl.dump(forecast_shop_list,open('save/forecast_shop_list.pkl','wb'))
        


if __name__ == '__main__':
    df_comment = pd.read_csv('t_comment.csv').fillna(0)
    df_ads = pd.read_csv('t_ads.csv').fillna(0)
    df_order = pd.read_csv('t_order.csv').fillna(0)
    df_product = pd.read_csv('t_product.csv').fillna(0)
    df_sales_sum = pd.read_csv('t_sales_sum.csv').fillna(0)
    df_Sales_Forecast = pd.read_csv('Sales_Forecast_Upload_Sample.csv').fillna(0)

    dl = Data_loader(df_comment,df_ads,df_order,df_product,df_sales_sum,df_Sales_Forecast)
    dl.create_train_data()
    dl.create_test_data()

