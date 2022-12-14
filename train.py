import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pong = pd.read_csv('pong_data copy.csv')
pong = pd.DataFrame(pong)

#print(pong.describe())

# sns.pairplot(pong)
# plt.show()


X_pong = pong.drop('paddle_direction', axis=1)
#print(X_pong.shape)

y_pong = pong['paddle_direction']
#print(y_pong.shape)


# Grab the column names
# print(pong.columns)


# from sklearn.mixture import GaussianMixture      # 1. Choose the model class
# model = GaussianMixture(n_components=3,
#             covariance_type='full')  # 2. Instantiate the model with hyperparameters
# model.fit(X_pong)                    # 3. Fit to data. Notice y is not specified!
# y_gmm = model.predict(X_pong)        # 4. Determine cluster labels
# pong['cluster'] = y_gmm
# sns.lmplot(data=pong, hue='paddle_direction',
#            col='cluster', fit_reg=False);




from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier

numeric_features = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy',
       'paddle_y', 'Ball.RADIUS', 'Paddle.L', 'Paddle.STEP', 'WIDTH', 'HEIGHT',
       'BORDER', 'VELOCITY', 'FPS']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(2))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        #('LRclassifier', LinearRegression(fit_intercept=True)),
                        ('SGDclassifier', SGDClassifier())])

X_train, X_test, y_train, y_test = train_test_split(X_pong, y_pong, test_size=0.2)

model.fit(X_train, y_train)
print("model TRAIN score: %.3f" % model.score(X_train, y_train))
print("model TEST score: %.3f" % model.score(X_test, y_test))

#Viz
# from sklearn import set_config
# set_config(display='diagram')
# print(model)

#see list of param:
#print(model.get_params().keys())

#more fun: tune up a model if you want

# param_grid = {
#     'preprocessor__num__imputer__strategy': ['mean', 'median'],
# #     'classifier__C': [0.1, 1.0, 10, 100],
#     'classifier__fit_intercept': [True, False],
#     'preprocessor__num__poly__degree': [1,2,3,4],
# }

# grid_search = GridSearchCV(model, param_grid, cv=3)
# grid_search.fit(X_train, y_train)

# print(("best TEST score from grid search: %.3f"
#        % grid_search.score(X_test, y_test)))


# #First, save test data:
X_test.to_csv('X_test.csv',index=False)
y_test.to_csv('y_test.csv',index=False)

# #then save model:
from joblib import dump, load
dump(model, 'mymodel.joblib') 


# # #Test: load and predict
# model2 = load('mymodel.joblib') 
# print(model2.score(X_test,y_test))
# print(model2.predict(X_test))




