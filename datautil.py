import pandas as pd
import numpy as np

class Data_loader():
	#建一个类用来加载某个店铺的某个月份统计信息
	def __init__(self,shop_id,month):
		self.shop_id = shop_id
		self.month = month
	def conver_comment(self,data_file='t_comment.csv'):
		df=pd.read_csv(data_file)
		#条件查询，得到这个商铺和这个月的信息
		df=df[np.logical_and(df.shop_id==self.shop_id,df.create_dt.str.contains(self.month))]		
		print (df)
		#返回统计数
		print (np.sum(df.bad_num),np.sum(df.cmmt_num),\
			np.sum(df.dis_num),np.sum(df.good_num),np.sum(df.mid_num))
	
	def conver_ads():
		pass

	def conver_order():
		pass

	def conver_product():
		pass
if __name__ == '__main__':
	dl=Data_loader(1494,'2017-01')
	dl.conver_comment()
	
	