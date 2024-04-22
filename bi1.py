import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv(r"C:\Users\chaim\Downloads\ATP.csv", encoding='utf-8')
head = data.head()

data.info()


df=data.drop(columns=['score','tourney_name','winner_name',
                      'loser_name','minutes', 'l_1stIn', 'l_svpt',
                      'l_SvGms','l_df', 'l_bpSaved',  'w_1stIn',  'w_SvGms',
                       'w_svpt', 'w_bpSaved','w_df','winner_rank_points', 'loser_rank_points'])
print("remained features ", df.columns)

numeric_columns = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht']
df[numeric_columns] = df[numeric_columns].astype(float)

df.tourney_date.head()

#"tourney_date" is in the format of YYYYMMDD
df['tourney_year'] = df.tourney_date.astype(str).str[:4].astype(int)
df['tourney_month'] = df.tourney_date.astype(str).str[4:6].astype(int)
#Now drop "tourney_date"
df = df.drop(columns=['tourney_date'])

df = df.rename(columns={"loser_age": "first_age", 
                        
                        "l_1stWon":"first_1stWon",
                        "l_2ndWon":"first_2ndWon" ,
                        "l_ace" :"first_ace",
                        "l_bpFaced" :"first_bpFaced",
                        
                        
                        "loser_hand": "first_hand",
                        "loser_ht": "first_ht", 
                        "loser_id": "first_id", 
                        "loser_ioc": "first_ioc",
                        "loser_rank": "first_rank", 
                        "loser_rank_points": "first_rank_points",
                        
                        "winner_age": "second_age", 
                        
                        "w_1stWon":"second_1stWon",
                        "w_2ndWon":"second_2ndWon" ,
                        "w_ace" :"second_ace",
                        "w_bpFaced" :"second_bpFaced",
                        
                        "winner_hand": "second_hand",
                        "winner_ht": "second_ht", 
                        "winner_id": "second_id", 
                        "winner_ioc": "second_ioc",
                        "winner_rank": "second_rank", 
                        "winner_rank_points": "second_rank_points",
                        
                       },)
# Visualizing the collelations between all variables of the data.
plt.figure(1 , figsize = (30,20))
cor = sns.heatmap(df.corr(), annot = True)
cor.set_title('Correlation Heatmap', fontdict={'fontsize':25},color="red" ,pad=12)
plt.show()

copy_2_df = df.copy()

copy_2_df[[ 'first_age','first_hand','first_ht','first_id','first_ioc','first_rank','first_1stWon','first_2ndWon','first_ace','first_bpFaced',
            'second_age','second_hand','second_ht','second_id','second_ioc','second_rank','second_1stWon','second_2ndWon','second_ace','second_bpFaced']]\
=copy_2_df[['second_age','second_hand','second_ht','second_id','second_ioc','second_rank','second_1stWon','second_2ndWon','second_ace','second_bpFaced',
             'first_age','first_hand','first_ht','first_id','first_ioc','first_rank','first_1stWon','first_2ndWon','first_ace','first_bpFaced']]

## Construct label feature
winner_player2 = np.zeros(df.shape[0]) # second player wins so label=0
df['label'] = winner_player2

winner_player1 = np.ones(copy_2_df.shape[0]) # first player wins so label=1
copy_2_df['label'] = winner_player1 

df = pd.concat([df,copy_2_df])
#shuffle data
# df = df.sample(frac=1) .reset_index(drop=True)
# df

hand_encoder = LabelEncoder()
df['first_hand'] =(df['first_hand'].astype(str))
df['second_hand'] = (df['second_hand'].astype(str))
df['first_ioc'] = df['first_ioc'].astype(str)
df['second_ioc'] = (df['second_ioc'].astype(str))
df['surface'] = (df['surface'].astype(str))
df['tourney_id'] = (df['tourney_id'].astype(str))

df.info()

df.head()


hand_encoder = LabelEncoder()
df['first_hand'] = hand_encoder.fit_transform(df['first_hand'])
df['second_hand'] = hand_encoder.transform(df['second_hand'])

df['first_ioc'] = LabelEncoder().fit_transform(df['first_ioc'])
df['second_ioc'] = LabelEncoder().fit_transform(df['second_ioc'])


lb = LabelBinarizer()
lb.fit(df['surface'])
df['surface'] = lb.transform(df['surface'])

df['surface'] = LabelBinarizer().fit_transform(df['surface'] )
df['tourney_id'] = LabelEncoder().fit_transform(df['tourney_id'])

df.info()

# Identify missing values
missing_values = df.isnull().sum()

# Remove columns with missing values above a certain threshold
missing_threshold = len(df) * 0.5
df = df.dropna(thresh=missing_threshold, axis=1)

# Impute missing values for numeric data using mean strategy
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Impute missing values for non-numeric data using most frequent strategy
non_numeric_cols = df.select_dtypes(include=['object']).columns
missing_values = df[non_numeric_cols].isnull().sum()
if missing_values.sum() > 0:
    imputer = SimpleImputer(strategy='most_frequent')
    df[non_numeric_cols] = imputer.fit_transform(df[non_numeric_cols])


print(df.head())

# Display final shape of the cleaned and preprocessed dataset
print('Final shape of data after cleaning and preprocessing: ', df.shape)


y = df['label']
df_X = df.drop(columns='label')

# split data : 80% for train and 20% for test.
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.1)

#Call the classifier
RF_classifier = RandomForestClassifier(n_estimators=100)
#fit the data
RF_classifier.fit(X_train, y_train)
#predict 
RF_predictions = RF_classifier.predict(X_test)

print('Confusion matrix')
print(confusion_matrix(y_test,RF_predictions))
print('Classification report')
print(classification_report(y_test,RF_predictions))
print('Accuracy= ', accuracy_score(y_test, RF_predictions))



#Call the classifier
XGB_classifier = XGBClassifier()
#fit the data
XGB_classifier.fit(X_train, y_train)
#predict 
XGB_predictions = XGB_classifier.predict(X_test)

print('Confusion matrix')
print(confusion_matrix(y_test,XGB_predictions))
print('Classification report')
print(classification_report(y_test,XGB_predictions))
print('Accuracy= ', accuracy_score(y_test, XGB_predictions))



KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train,y_train)
score = KNN_model.score(X_test,y_test)


print("KNN score: ", score)






svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred )
svm_f1_score = f1_score(y_test, svm_pred )
svm_auc = roc_auc_score(y_test, svm_pred )
svm_recall = recall_score(y_test,svm_pred )

print(classification_report(y_test,svm_pred ))


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print(classification_report(y_test,rf_preds))


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print(classification_report(y_test,dt_preds))


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)


print(classification_report(y_test,xgb_preds))


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_pred)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_preds)
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_preds)
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_preds)
plt.plot(svm_fpr, svm_tpr, label='SVM')
plt.plot(rf_fpr, rf_tpr, label='Random Forest')
plt.plot(dt_fpr, dt_tpr, label='Decision Tree')
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

import pickle

# Save the svm regression model to disk
svm_filename = 'svm_model.pkl'
pickle.dump(svm, open(svm_filename, 'wb'))

# Save the decision tree model to disk
dt_filename = 'dt_model.pkl'
pickle.dump(dt_model, open(dt_filename, 'wb'))

# Save the random forest model to disk
rf_filename = 'rf_model.pkl'
pickle.dump(rf_model , open(rf_filename, 'wb'))

# Save the gradient boosting model to disk
gb_filename = 'gb_model.pkl'
pickle.dump(XGB_classifier, open(gb_filename, 'wb'))