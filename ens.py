#kfold from sklearn
#use to create prediction probabilities for each data sample for each model type so we can ensemble
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

#create leave-one-out probability predictions for each model
for train_index,test_index in kf.split(X):
    models = [log_reg_model, svm_model, sgd_model, knn_model, gpc_model, gnb_model, tree_model, rf_model, gbc_model, ada, nn_model]
    for model in models:
        print(type(model).__name__)
        model.fit(X.loc[train_index],y.loc[train_index])
        df.loc[test_index, type(model).__name__]=model.predict(X.loc[test_index])
        
#create and predict ensemble model
#linear regression with sklearn
meta_X,meta_y = df.iloc[:,-NUM_MODELS:-NUM_OUTPUTS], df.iloc[:,-NUM_OUTPUTS:]
#split dataset into train and test data
meta_x_train, meta_x_test, meta_y_train, meta_y_test = train_test_split(meta_X, meta_y, test_size=0.2, random_state=1, stratify=y)

meta_model = LinearRegression()
meta_model.fit(meta_x_train, meta_y_train)

meta_preds = meta_model.predict(meta_x_test)
#use rmse instead
meta_score = roc_auc_score(meta_y_test, meta_preds)
print(meta_score)
