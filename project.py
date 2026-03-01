# You can run this script, but keep in mind that you will receive a lot of popups from all the plots, so I don't recommend running it this way.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report,
    roc_curve, 
    auc
)
from sklearn import metrics
import os
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
import random
import warnings
import xgboost
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
np.random.seed(34)
warnings.filterwarnings('ignore')


index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names


dftrain = pd.read_csv('train_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv('test_FD001.txt',sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv('RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])
dfvalid.shape

train = dftrain.copy()
valid = dfvalid.copy()

train

print('Shape of the train dataset : ',train.shape)
print('Shape of the validation dataset : ',valid.shape)
print('Percentage of the validation dataset : ',len(valid)/(len(valid)+len(train)))

print('Total None values in the train dataset : ',train.isna().sum())

train.loc[:,['unit_number','time_cycles']].describe()

train.loc[:,'s_1':].describe().transpose()

max_time_cycles=train[index_names].groupby('unit_number').max()
plt.figure(figsize=(20,50))
ax=max_time_cycles['time_cycles'].plot(kind='barh',width=0.8, stacked=True,align='center')
plt.title('Turbofan Engines LifeTime',fontweight='bold',size=30)
plt.xlabel('Time cycle',fontweight='bold',size=20)
plt.xticks(size=15)
plt.ylabel('unit',fontweight='bold',size=20)
plt.yticks(size=15)
plt.grid(True)
plt.tight_layout()
plt.show()

sns.displot(max_time_cycles['time_cycles'],kde=True,bins=20,height=6,aspect=2)
plt.xlabel('max time cycle')

def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit_number')
    max_time_cycles = train_grouped_by_unit['time_cycles'].max()
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1)
    return merged

train = add_RUL_column(train)

train[['unit_number','RUL']]

maxrul_u = train.groupby('unit_number').max().reset_index()
maxrul_u.head()

# Compute the correlation matrix
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(230, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

Sensor_dictionary={}
dict_list=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]
i=1
for x in dict_list :
    Sensor_dictionary['s_'+str(i)]=x
    i+=1
Sensor_dictionary

def plot_signal(df, Sensor_dic, signal_name):
    plt.figure(figsize=(13,5))
    for i in df['unit_number'].unique():
        if (i % 10 == 0):   #For a better visualisation, we plot the sensors signals of 20 units only
            plt.plot('RUL', signal_name, data=df[df['unit_number']==i].rolling(10).mean())

    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 300, 25))
    plt.ylabel(Sensor_dic[signal_name])
    plt.xlabel('Remaining Useful Life')
    plt.show()


for i in range(1,22):
    try:
        plot_signal(train, Sensor_dictionary,'s_'+str(i))
    except:
        pass

for x in sensor_names:
    plt.figure(figsize=(13,7))
    plt.boxplot(train[x])
    plt.title(x)
    plt.show()

train.loc[:,'s_1':].describe().transpose()

drop_labels = index_names+setting_names
X_train=train.drop(columns=drop_labels).copy()
X_train, X_test, y_train, y_test=train_test_split(X_train,X_train['RUL'], test_size=0.3, random_state=42)

scaler = MinMaxScaler()
#Dropping the target variable
X_train.drop(columns=['RUL'], inplace=True)
X_test.drop(columns=['RUL'], inplace=True)
#Scaling X_train and X_test
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.fit_transform(X_test)
#Conserve only the last occurence of each unit to match the length of y_valid
X_valid = valid.groupby('unit_number').last().reset_index().drop(columns=drop_labels)
#scaling X_valid
X_valid_s=scaler.fit_transform(X_valid)

print(X_valid_s.shape)
print(y_valid.shape)

sensor_names=['s_{}'.format(i) for i in range(1,22) if i not in [1,5,6,10,16,18,19]]
pd.DataFrame(X_train_s,columns=['s_{}'.format(i) for i in range(1,22)])[sensor_names].hist(bins=100, figsize=(18,16))

rf = RandomForestRegressor(max_features="sqrt", random_state=42)

#plot real data and the predicted one to make some comparison
def plot_predActual(y_test, y_test_hat):
    indices = np.arange(len(y_test_hat))
    wth= 0.6
    plt.figure(figsize=(70,30))
    true_values = [int(x) for x in y_test.values]
    predicted_values = list(y_test_hat)

    plt.bar(indices, true_values, width=wth,color='b', label='True RUL')
    plt.bar([i for i in indices], predicted_values, width=0.5*wth, color='r', alpha=0.7, label='Predicted RUL')

    plt.legend(prop={'size': 40})
    plt.tick_params(labelsize=40)

    plt.show()

regression_results = []

def evaluate(y_true, y_pred, label='test', model_name=None, results_list=None):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - {label} set RMSE: {rmse:.3f}, R²: {r2:.3f}")

    if results_list is not None:
        results_list.append((model_name, label, rmse, r2))

lr=LinearRegression() 
lr.fit(X_train_s, y_train) 

y_lr_train = lr.predict(X_train_s) 
evaluate(y_train,y_lr_train, label='train')

y_lr_test = lr.predict(X_test_s)  
evaluate(y_test, y_lr_test, label='test')

y_lr_valid= lr.predict(X_valid_s) 
evaluate(y_valid, y_lr_valid, label='valid')

rf = RandomForestRegressor(max_features="sqrt", random_state=42)
rf.fit(X_train_s, y_train)

y_rf_train = rf.predict(X_train_s)
evaluate(y_train,y_rf_train, label='train')
y_rf_test = rf.predict(X_test_s)
evaluate(y_test, y_rf_test, label='test')
y_rf_valid = rf.predict(X_valid_s)
evaluate(y_valid, y_rf_valid, label='valid')

plot_predActual(y_valid, y_rf_valid)

print('maximum of y_train : ',y_train.max())
print('maximum of y_test : ',y_test.max())
print('maximum of y_valid : ',y_valid.max())

plt.hist(y_test)

drop_labels2=['s_1', 's_5','s_6','s_10', 's_16', 's_18', 's_19']

# drop the constant columns from the train dataset
X_train_2=X_train.drop(columns=drop_labels2, axis=1) 
# drop the constant columns from the test dataset
X_test_2=X_test.drop(columns=drop_labels2, axis=1)  

X_train_2_s=scaler.fit_transform(X_train_2) 
X_test_2_s=scaler.fit_transform(X_test_2) 
y_train_clip=y_train.clip(upper=195)  

# drop the constant columns from the validation dataset
X_valid_2=X_valid.drop(columns=drop_labels2, axis=1)  
X_valid_2_s=scaler.fit_transform(X_valid_2) 

lr=LinearRegression()
lr.fit(X_train_2_s, y_train_clip)

y_lr_train = lr.predict(X_train_2_s)
evaluate(y_train_clip,y_lr_train, label='train')

y_lr_test = lr.predict(X_test_2_s)
evaluate(y_test, y_lr_test, label='test')

y_lr_valid = lr.predict(X_valid_2_s)
evaluate(y_valid, y_lr_valid, label='valid')

rf.fit(X_train_2_s, y_train_clip)
# predict and evaluate
y_rf_train = rf.predict(X_train_2_s)
evaluate(y_train_clip,y_rf_train, label='train')

y_rf_test = rf.predict(X_test_2_s)
evaluate(y_test, y_rf_test, label='test')

y_rf_valid = rf.predict(X_valid_2_s)
evaluate(y_valid, y_rf_valid, label='valid')

xgb = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.02, gamma=0, subsample=0.8,colsample_bytree=0.5, max_depth=3)
xgb.fit(X_train_2_s, y_train_clip)


y_xgb_train = xgb.predict(X_train_2_s)
evaluate(y_train_clip,y_xgb_train, label='train')

y_xgb_test = xgb.predict(X_test_2_s)
evaluate(y_test, y_xgb_test, label='test')

y_xgb_valid = xgb.predict(X_valid_2_s)
evaluate(y_valid, y_xgb_valid, label='valid')

plot_predActual(y_valid, y_rf_valid)

df=train.copy()

for x in X_train_2.columns:
    df[x+'_rm']=0

df.columns

drop_labels2=['s_1', 's_5','s_6','s_10',  's_16', 's_18', 's_19']
df=df.drop(columns=setting_names+drop_labels2+['RUL'], axis=1)

X_valid_3=valid.drop(columns=index_names+setting_names+drop_labels2, axis=1)

def update_rolling_mean(data, mask):
    for x, group in mask.groupby("unit_number"):
        for x in X_train_2.columns:
            data.loc[group.index[10:], x+"_rm"] = data.loc[group.index, x].rolling(10).mean()[10:]
            data.loc[group.index[:10], x+"_rm"] = data.loc[group.index[:10], x]

update_rolling_mean(df, df)
update_rolling_mean(X_valid_3, valid)

X_valid_3=X_valid_3.fillna(0)

# Dealing with last line problem
df.iloc[-1,-14:]=df.iloc[-2,-14:]
X_valid_3.iloc[-1,-14:]=X_valid_3.iloc[-2,-14:]

train_tm=df

train_tm=train_tm.drop(columns=index_names, axis=1)

X_train_tm, X_test_tm, y_train_tm, y_test_tm=train_test_split(train_tm,train['RUL'].clip(upper=195), test_size=0.35, random_state=42)
X_train_tm_s=scaler.fit_transform(X_train_tm)
X_test_tm_s=scaler.fit_transform(X_test_tm)
X_val3=pd.concat([valid['unit_number'],X_valid_3],axis=1)
X_valid3 = X_val3.groupby('unit_number').last().reset_index().drop(columns=['unit_number'])
X_valid_s=scaler.fit_transform(X_valid3)

lr=LinearRegression()
lr.fit(X_train_tm_s, y_train_tm)

preds = [(X_train_tm_s, y_train_tm, 'train'),
         (X_test_tm_s, y_test_tm, 'test'),
         (X_valid_s, y_valid, 'valid')]

for X, y, lbl in preds:
    y_pred = lr.predict(X)
    evaluate(y, y_pred, lbl, 'Linear Regression', regression_results)

ridge = Ridge()
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}
ridge_grid = GridSearchCV(ridge, param_grid_ridge, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid.fit(X_train_tm_s, y_train_tm)

print("Best Ridge Params:", ridge_grid.best_params_)
ridge_best = ridge_grid.best_estimator_

# Predict and evaluate
for X, y, lbl in preds:
    y_pred = ridge_best.predict(X)
    evaluate(y, y_pred, lbl, 'Ridge Regression', regression_results)

rf=RandomForestRegressor(n_estimators=90,  max_depth=10, n_jobs=-1, max_features="sqrt",random_state=42)
rf.fit(X_train_tm_s, y_train_tm)
# predict and evaluate
for X, y, lbl in preds:
    y_pred = rf.predict(X)
    evaluate(y, y_pred, lbl, 'Random Forest Regressor', regression_results)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.06, 0.08, 0.1],
    'max_depth': [5, 6, 7]
}

xgb_grid = GridSearchCV(estimator=xgb_model,
                        param_grid=param_grid_xgb,
                        scoring='neg_mean_squared_error',
                        cv=3,
                        n_jobs=-1,
                        verbose=1)

xgb_grid.fit(X_train_tm_s, y_train_tm)
print("Best LightGBM Params:", xgb_grid.best_params_)
xgb_best = xgb_grid.best_estimator_
for X, y, lbl in preds:
    y_pred = xgb_best.predict(X)
    evaluate(y, y_pred, lbl, 'XGBoost Regressor', regression_results)

lgb_model = lgb.LGBMRegressor(random_state=42)
param_grid_lgb = {
    'n_estimators': [100],
    'learning_rate': [0.05, 0.07],
    'max_depth': [5, 7],
    'num_leaves': [20]
}

lgb_grid = GridSearchCV(lgb_model, param_grid_lgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
lgb_grid.fit(X_train_tm_s, y_train_tm)

print("Best LightGBM Params:", lgb_grid.best_params_)
lgb_best = lgb_grid.best_estimator_

for X, y, lbl in preds:
    y_pred = lgb_best.predict(X)
    evaluate(y, y_pred, lbl, 'LightGBM Regressor', regression_results)

regression_results

regression_df = pd.DataFrame(regression_results, columns=['Model', 'Dataset', 'RMSE', 'R2'])
regression_pivot = regression_df.pivot(index='Model', columns='Dataset', values=['RMSE', 'R2'])
print("\n=== Regression Model Performance Summary ===")
print(regression_pivot.round(3))

train_tm

train_tm_cv=train_tm.copy()
y=train['RUL'].clip(upper=195)
scores_train=[]
scores_test=[]
scores_validation=[]

xgb_best.fit(X_train_tm_s, y_train_tm)
plt.barh(train_tm.columns, xgb_best.feature_importances_)

model = LinearRegression()
model.fit(X=X_train_tm_s, y=y_train_tm)
plt.barh(X_train_tm.columns, model.coef_)

rf.fit(X_train_tm_s, y_train_tm)
plt.barh(train_tm.columns, rf.feature_importances_)

dftm= pd.concat([train['unit_number'],train_tm,train['RUL']],axis=1)

dftm

def plot_signal(df, signal_name):
    plt.figure(figsize=(13,5))
    for i in df['unit_number'].unique():
        if (i % 10 == 0):
            plt.plot('RUL', signal_name, data=df[df['unit_number']==i].rolling(8).mean())

    plt.ylabel(signal_name)
    plt.xlabel('Remaining Useful Life')
    plt.show()

for i in range(1,22):
     if i not in [1,5,6,10,16,18,19] :
        try:
            plot_signal(dftm, 's_'+str(i)+'_rm')
        except:
            pass

#-- fixing bins width -----
# Creating histogram of 2 bins(same width)
fig, ax = plt.subplots(figsize =(10, 7))
m=ax.hist(list(train['RUL']), bins = 2, edgecolor='black')
# Show plot
plt.show()
print(m)

# Creating histogram of 3 bins(same width)
fig, ax = plt.subplots(figsize =(10, 7))
m=ax.hist(list(train['RUL']), bins = 3, edgecolor='black')
# Show plot
plt.show()
print(m)

# Creating histogram of 4 bins(same width)
fig, ax = plt.subplots(figsize =(10, 7))
m=ax.hist(list(train['RUL']), bins = 4, edgecolor='black')
# Show plot
plt.show()
print(m)

l=len(list(train['RUL']))
k=l/4

#define function to calculate equal-frequency bins, bins=2
def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

#create histogram with equal-frequency bins
n, bins, patches = plt.hist(list(train['RUL']), equalObs(list(train['RUL']), 2), edgecolor='black')
plt.show()
print(bins)

#define function to calculate equal-frequency bins, bins=3
def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

#create histogram with equal-frequency bins
n, bins, patches = plt.hist(list(train['RUL']), equalObs(list(train['RUL']), 3), edgecolor='black')
plt.show()
print(bins)


#define function to calculate equal-frequency bins, bins=4
def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

#create histogram with equal-frequency bins
n, bins, patches = plt.hist(list(train['RUL']), equalObs(list(train['RUL']), 4), edgecolor='black')
plt.show()
print(bins)

y_train_tm


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true)))

# Mapping for risk categories
label_map = {0: "RISK ZONE", 1: "MODERATED RISK", 2: "NO RISK"}

# Transform RUL into categorical classes
y = []
for k in dftm['RUL']:
    if k <= 68:
        y.append(0)  
    elif k > 69 and k <= 137:
        y.append(1)  
    else:
        y.append(2)  

# Split data for classification
X_train_tm_c, X_test_tm_c, y_train_tm_c, y_test_tm_c = train_test_split(
    dftm.drop(columns=['unit_number', 'RUL']), 
    np.array(y), 
    test_size=0.35, 
    stratify=np.array(y)
)

# Scale the training and test data
X_train_tm_cs = scaler.fit_transform(X_train_tm_c)
X_test_tm_cs = scaler.fit_transform(X_test_tm_c)

y_valid_c = []
for k in y_valid['RUL']:
    if k <= 68:
        y_valid_c.append(0)  
    elif k > 69 and k <= 137:
        y_valid_c.append(1)  
    else:
        y_valid_c.append(2) 
y_valid_c = np.array(y_valid_c)

# Create and train XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,  # 3 classes + 1 (XGBoost counts from 0)
    random_state=42
)

# Parameter search for XGBoost classifier
param_grid_xgb_clf = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

xgb_clf_grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid_xgb_clf,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Fit the grid search model
xgb_clf_grid.fit(X_train_tm_cs, y_train_tm_c)
print("Best XGBoost Classification Params:", xgb_clf_grid.best_params_)

xgb_clf_best = xgb_clf_grid.best_estimator_
print("XGBoost Classes:", xgb_clf_best.classes_)
print("Label Map Keys:", label_map.keys())


xgb_clf_best = xgb_clf_grid.best_estimator_

# Training predictions
y_xgb_train = xgb_clf_best.predict(X_train_tm_cs)

# Test predictions
y_xgb_test = xgb_clf_best.predict(X_test_tm_cs)

# Create confusion matrix for test data
cm = confusion_matrix(y_test_tm_c, y_xgb_test, labels=xgb_clf_best.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=[label_map[c] for c in xgb_clf_best.classes_]
)
disp.plot()
plt.title('XGBoost Classification - Test Set')
plt.show()

print('XGBoost Classification')
print("Accuracy score of training %.3f" % metrics.accuracy_score(y_train_tm_c, y_xgb_train))
print("Error rate of training %.3f" % mean_absolute_percentage_error(y_train_tm_c, y_xgb_train))
print("Accuracy score of test %.3f" % metrics.accuracy_score(y_test_tm_c, y_xgb_test))
print("Error rate of test %.3f" % mean_absolute_percentage_error(y_test_tm_c, y_xgb_test))
print(metrics.classification_report(y_test_tm_c, y_xgb_test))

clf=RandomForestClassifier(n_estimators=5)

clf.fit(X_train_tm_cs,np.array(y_train_tm_c))
y_rfc_train=clf.predict(X_train_tm_cs)
y_rfc_test=clf.predict(X_test_tm_cs)
cm= confusion_matrix(y_test_tm_c, y_rfc_test, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_map[clf.classes_[0]],label_map[clf.classes_[1]],label_map[clf.classes_[2]]])
disp.plot()
plt.show()

# Measure the performance
print('Random Forest Classifier')
print("Accuracy score of test %.3f" %metrics.accuracy_score(y_train_tm_c, y_rfc_train))
print("Error rate of test %.3f" %mean_absolute_percentage_error(y_train_tm_c,y_rfc_train))

print("Accuracy score of test %.3f" %metrics.accuracy_score(y_test_tm_c, y_rfc_test))
print("Error rate of test %.3f" %mean_absolute_percentage_error(y_test_tm_c,y_rfc_test))

print(metrics.classification_report(y_test_tm_c,y_rfc_test))

y_rfc_valid=clf.predict(X_valid_s)
cm= confusion_matrix(y_valid_c, y_rfc_valid, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_map[clf.classes_[0]],label_map[clf.classes_[1]],label_map[clf.classes_[2]]])
disp.plot()
plt.show()
print('Random Forest Classifier')
print("Accuracy score of validation %.3f" %metrics.accuracy_score(y_valid_c, y_rfc_valid))
print("Error rate of validation %.3f" %mean_absolute_percentage_error(y_valid_c,y_rfc_valid))
print(metrics.classification_report(y_valid_c,y_rfc_valid))

print('XGBoost Classification - Validation')
print("Accuracy score of validation %.3f" % metrics.accuracy_score(y_valid_c, y_xgb_valid))
print("Error rate of validation %.3f" % mean_absolute_percentage_error(y_valid_c, y_xgb_valid))
print(metrics.classification_report(y_valid_c, y_xgb_valid))

plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_clf_best, max_num_features=20)
plt.title('XGBoost Feature Importance for Classification')
plt.show()

def transform_to_classes(d):
    y = []
    for k in d:
        if k <= 68:
            y.append(0)  
        elif k > 69 and k <= 137:
            y.append(1)  
        else:
            y.append(2) 
    return y

# K-fold cross-validation
train_tm_cv = dftm.drop(columns=['unit_number', 'RUL']).copy()
y_rul = dftm['RUL']

print("\nPerforming 4-fold cross-validation with XGBoost classifier:")
cv = KFold(n_splits=4, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in cv.split(train_tm_cv):
    print(f"\nFold {fold}:")
    X_train = train_tm_cv.iloc[train_index, :]
    X_test = train_tm_cv.iloc[test_index, :]
    y_train = np.array(transform_to_classes(y_rul[train_index]))
    y_test = np.array(transform_to_classes(y_rul[test_index]))
    
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a smaller XGBoost model for CV
    cv_xgb = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    
    cv_xgb.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = cv_xgb.predict(X_train_scaled)
    y_test_pred = cv_xgb.predict(X_test_scaled)
    
    # Performance metrics
    print(f"Training accuracy: {metrics.accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Test accuracy: {metrics.accuracy_score(y_test, y_test_pred):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=cv_xgb.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[label_map[c] for c in cv_xgb.classes_]
    )
    disp.plot()
    plt.title(f'XGBoost Classification - Fold {fold} Test Set')
    plt.show()
    
    fold += 1

# Compare with other classifiers (RF, NB, KNN)
print("\nComparing XGBoost with other classifiers:")

# Need to redefine xgb_clf as it was giving zero validation accuracy for some reason
xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )

xgb_clf.fit(X_train_tm_cs, y_train_tm_c)
y_xgb_test = xgb_clf.predict(X_test_tm_cs)
y_xgb_valid = xgb_clf.predict(X_valid_s)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_tm_cs, y_train_tm_c)
y_rf_test = rf_clf.predict(X_test_tm_cs)
y_rf_valid = rf_clf.predict(X_valid_s)

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train_tm_cs, y_train_tm_c)
y_nb_test = nb_clf.predict(X_test_tm_cs)
y_nb_valid = nb_clf.predict(X_valid_s)

# KNN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_tm_cs, y_train_tm_c)
y_knn_test = knn_clf.predict(X_test_tm_cs)
y_knn_valid = knn_clf.predict(X_valid_s)

# Define classifiers and predictions
classifiers = ['XGBoost', 'Random Forest', 'Naive Bayes', 'KNN']
test_preds = [y_xgb_test, y_rf_test, y_nb_test, y_knn_test]
valid_preds = [
    y_xgb_valid,
    y_rf_valid,
    y_nb_valid,
    y_knn_valid
]

print("\n=== Test Set Performance ===")
for i, clf in enumerate(classifiers):
    acc = metrics.accuracy_score(y_test_tm_c, test_preds[i])
    print(f"{clf} - Accuracy: {acc:.3f}")

print("\n=== Validation Set Performance ===")
for i, clf in enumerate(classifiers):
    acc = metrics.accuracy_score(y_valid_c, valid_preds[i])
    print(f"{clf} - Accuracy: {acc:.3f}")


# For XGBoost
xgb_probs = xgb_clf_best.predict_proba(X_test_tm_cs)
for i in range(3):  # 3 classes
    fpr, tpr, _ = metrics.roc_curve(y_test_tm_c == (i), xgb_probs[:, i])
    plt.plot(fpr, tpr, label=f'XGBoost - {label_map[i]} (AUC = {metrics.auc(fpr, tpr):.2f})')

plt.title('ROC Curves for XGBoost Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

regression_results_edited = [row[:-1] for row in regression_results]

regression_df = pd.DataFrame(regression_results_edited, columns=['Model', 'Dataset', 'RMSE'])
regression_pivot = regression_df.pivot(index='Model', columns='Dataset', values=['RMSE'])
print("\n=== Regression Model Performance Summary ===")
print(regression_pivot.round(3))


