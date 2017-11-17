import math
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import  make_scorer

feature_all = pkl.load(open('feature_list.pkl','rb'))
feature_all = np.array(feature_all)
feature_all=np.nan_to_num(feature_all)

lable_all = pkl.load(open('label_list.pkl','rb'))

def my_custom_loss_func(ground_truth, predictions):
	#自定义评分函数
	differ = list(map(lambda x: x[0]-x[1],zip(ground_truth,predictions)))
	abs_differ = map(abs,differ)
	wmae_score = sum(abs_differ)/sum(ground_truth)
	return wmae_score

def find_best_parameters():
	score = make_scorer(my_custom_loss_func, greater_is_better=False)
	clf=Pipeline([('ss',StandardScaler()),('GBR',GradientBoostingRegressor())])
	parameters={'GBR__learning_rate':[0.1],'GBR__n_estimators':list((range(20,301,30))),'GBR__max_depth':[4]}

	gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=5,scoring=score)
	gs.fit(feature_all,lable_all)
	print(gs.best_params_,gs.best_score_)


def test_result():
	#测试并保存预测结果
	ss = StandardScaler()
	feature_ss = ss.fit_transform(feature_all)
	X_train,X_test,y_train,y_test = train_test_split(feature_ss,lable_all,\
	test_size=0.25,random_state=33)

	GBR = GradientBoostingRegressor(learning_rate=0.1,n_estimators=300,max_depth=4)
	score = make_scorer(my_custom_loss_func, greater_is_better=False)
	cv_scores = cross_val_score(GBR, feature_all, lable_all, cv=5, scoring=score)
	print ("GBR train score is {} mean {}".format(cv_scores,np.mean(cv_scores)))


def save_data():
	forecast_feature_list = pkl.load(open('forecast_feature_list.pkl','rb'))
	forecast_feature = np.array(forecast_feature_list)
	forecast_feature=np.nan_to_num(forecast_feature)
	forecast_shop_list = pkl.load(open('forecast_shop_list.pkl','rb'))

	GBR = GradientBoostingRegressor(learning_rate=0.1,n_estimators=300,max_depth=4)
	GBR.fit(feature_all,lable_all)
	ss = StandardScaler()
	forecast_feature_ss = ss.fit_transform(forecast_feature)
	forecast_pred = GBR.predict(forecast_feature_ss)
	ans = pd.DataFrame(forecast_pred,forecast_shop_list)
	ans.to_csv('ans.csv')
	print('done')

if __name__ == '__main__':
	find_best_parameters()