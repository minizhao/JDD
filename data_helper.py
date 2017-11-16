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

    def conver_comment(self, df):
        # df = pd.read_csv(data_file)
        # 条件查询，得到这个商铺和这个月的信息
        df = df[np.logical_and(df.shop_id == self.shop_id, df.create_dt.str.contains(self.month))]
        # print(df)
        # 返回统计数
        return [np.sum(df.bad_num), np.sum(df.cmmt_num), \
              np.sum(df.dis_num), np.sum(df.good_num), np.sum(df.mid_num)]

    def conver_ads(self,df):
        # df = pd.read_csv(data_file)
        df = df[np.logical_and(df.shop_id == self.shop_id,df.create_dt.str.contains(self.month))]
        # print (df)
        return[np.sum(df.charge),np.sum(df.consume),np.sum(df.shop_id)]
    #
    def conver_order(self,df):
        # df = pd.read_csv(data_file)
        df = df[np.logical_and(df.shop_id == self.shop_id,df.ord_dt.str.contains(self.month))]
        # print(df)
        return [np.sum(df.sale_amt),np.sum(df.offer_amt),np.sum(df.offer_cnt),\
              np.sum(df.rtn_cnt),np.sum(df.rtn_amt),np.sum(df.ord_cnt),len(df.pid.drop_duplicates()),\
              np.sum(df.user_cnt)]
    #
    def conver_product(self,df):
        # df = pd.read_csv(data_file)
        df = df[df.shop_id == self.shop_id]
        # return (df)
        return [len(df.brand.drop_duplicates()),len(df.cate.drop_duplicates()),len(df.pid.drop_duplicates())]

    def create_train_data(self):
        feature_list=[]
        label_list=[]
        all_shop=self.df_ads.shop_id.drop_duplicates()
        months=list(set([x[:7] for x in self.df_ads.create_dt.drop_duplicates()]))#提取月份
        for shop_id in all_shop:
            self.shop_id=shop_id
            for month in months:
                self.month=month
                feature = np.concatenate((dl.conver_comment(df_comment), dl.conver_ads(df_ads),\
                                      dl.conver_order(df_order), dl.conver_product(df_product)))
                label = df_sales_sum[np.logical_and(df_sales_sum.shop_id==shop_id,df_sales_sum.dt.str.contains(month))]
                
                if label.shape[0]!=0:
                    feature_list.append(feature)
                    label_list.append(label.sale_amt_3m.tolist()[0])
                    print(str(len(label_list))+'/'+str(len(months)*len(all_shop)))
        
        pkl.dump(feature_list,open('feature_list.pkl','wb'))
        pkl.dump(label_list,open('label_list.pkl','wb'))


    def create_test_data(self):
        forecast_feature_list=[]
        forecast_shop_list=[]
        all_forecast_shop=self.df_Sales_Forecast.shop_id.drop_duplicates()
        for shop_id in all_forecast_shop:
            self.shop_id=shop_id
            self.month='2017-04'
            
            feature = np.concatenate((dl.conver_comment(df_comment), dl.conver_ads(df_ads),\
                                      dl.conver_order(df_order), dl.conver_product(df_product)))
                
            forecast_feature_list.append(feature)
            forecast_shop_list.append(shop_id)
            
            print(str(len(forecast_feature_list))+'/'+str(len(all_forecast_shop)))
        
        assert len(feature_list)==len(all_forecast_shop)
        pkl.dump(forecast_feature_list,open('forecast_feature_list.pkl','wb'))
        pkl.dump(forecast_shop_list,open('forecast_shop_list.pkl','wb'))


if __name__ == '__main__':
    df_comment = pd.read_csv('t_comment.csv')
    df_ads = pd.read_csv('t_ads.csv')
    df_order = pd.read_csv('t_order.csv')
    df_product = pd.read_csv('t_product.csv')
    df_sales_sum = pd.read_csv('t_sales_sum.csv')
    df_Sales_Forecast = pd.read_csv('Sales_Forecast_Upload_Sample.csv')

    dl = Data_loader(df_comment,df_ads,df_order,df_product,df_sales_sum,df_Sales_Forecast)
    dl.create_test_data()

