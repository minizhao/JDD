import math
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import 


feature_all = pkl.load(open('feature_list.pkl','rb'))
lable_all = pkl.load(open('label_list.pkl','rb'))


feature_all = np.array(feature_all)
where_are_nan = np.isnan(feature_all)
feature_all[where_are_nan]=0
# print (feature_all[0])
ss = StandardScaler()
feature_ss = ss.fit_transform(feature_all)
X_train,X_test,y_train,y_test = train_test_split(feature_ss,lable_all,\
	test_size=0.25,random_state=0)
# n_estimators = np.linspace(100,10000,100)
# print (n_estimators)

# max_depth = [3,4,5,6,7,8,9,10]

# learning_rate=np.linspace(0.1,1,8)
# # for n_e in n_estimators:
# for md in max_depth:
# for lr in learning_rate:
GBR = GradientBoostingRegressor(learning_rate=0.13,n_estimators=350,max_depth=5)
GBR = GBR.fit(X_train,y_train)

pred = GBR.predict(X_test)


# result = zip(pred,y_test)
# for pred_value,label_value in result:
# 	print (pred_value,'\t',label_value)
differ = list(map(lambda x: x[0]-x[1],zip(y_test,pred)))
# print(differ)
abs_differ = map(abs,differ)

wmae_score = sum(abs_differ)/sum(y_test)

print (wmae_score)
print ("GBR train score is {} ".format(wmae_score))


forecast_feature_list = pkl.load(open('forecast_feature_list.pkl','rb'))
forecast_shop_list = pkl.load(open('forecast_shop_list.pkl','rb'))
forecast_feature = np.array(forecast_feature_list)
where_are_nan = np.isnan(forecast_feature)
forecast_feature[where_are_nan]=0

ss = StandardScaler()
feature_ss = ss.fit_transform(feature_all)
forecast_feature_ss = ss.transform(forecast_feature)



GBR = GradientBoostingRegressor(learning_rate=0.16,n_estimators=110,max_depth=3)
GBR = GBR.fit(feature_ss,lable_all)
forecast_pred = GBR.predict(forecast_feature_ss)
# print (len(forecast_pred))


# print(len(forecast_shop_list))

ans = pd.DataFrame(forecast_pred,forecast_shop_list)


# print(ans)

ans.to_csv('ans.csv',header=False)
