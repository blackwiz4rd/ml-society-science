import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sklearn.utils

df = pd.read_csv('https://pkgstore.datahub.io/machine-learning/dermatology/dermatology_csv/data/8a7c88e486ea2cb227adc99d0de841d1/dermatology_csv.csv')
df = df.fillna(0)
#print(df, df.shape)
print(df.head())

disease_class = ['psoriasis', ' seboreic_dermatitis', 'lichen_planus', 'pityriasis_rosea', 'cronic_dermatitis', ' pityriasis_rubra_pilaris']

for c, d in df.groupby('class'):
    d.age.plot.hist(label=c, alpha=0.6)
plt.legend()
plt.xlabel('Age content')
plt.title('Age content per class')
plt.show()

train_data, test_data = train_test_split(df, test_size = .2)

print(train_data)
# print(train_data.columns[:-1].values)
# print(train_data[['class']])

# for feature in train_data.columns[:-1].values:
	# print(train_data[[feature]], train_data['class'])	
	# print(np.max(train_data['class']), np.max(train_data[[feature]]))
	# KNeighborsClassifier(n_neighbors=5).fit(train_data[[feature]], train_data['class'])

features = train_data.columns[:-1]

models = [KNeighborsClassifier(n_neighbors=5).fit(train_data[[f]], train_data['class']) for f in features]

train_scores = [accuracy_score(train_data['class'], m.predict(train_data[[f]])) for m, f in zip(models, features)]
plt.barh(range(len(features)), train_scores)
plt.yticks(range(len(features)), features)
plt.gcf().set_size_inches(10, 5)
plt.tight_layout()
plt.show()

scores = [accuracy_score(test_data['class'], m.predict(test_data[[f]])) for m, f in zip(models, features)]
plt.barh(range(len(features)), scores)
plt.yticks(range(len(features)), features)
plt.gcf().set_size_inches(10, 5)
plt.show()

# rescale features
df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)
df_scaled['class'] = df['class']

train_data_s, test_data_s = train_test_split(df_scaled, test_size=0.2)

N, _ = train_data_s.shape
N_test, _ = test_data_s.shape

ks = range(1, 100)

models = [KNeighborsClassifier(n_neighbors=k).fit(train_data_s[features], train_data_s['class']) for k in ks]

train_scores = [accuracy_score(train_data_s['class'], m.predict(train_data_s[features])) for m in models]

test_scores = [accuracy_score(test_data_s['class'], m.predict(test_data_s[features])) for m in models]

plt.semilogx(ks, train_scores, ks, test_scores);
plt.legend(["Train", "Test"])
plt.xlabel('ks')
plt.ylabel('score')
plt.show()

neighbor_ks = range(1, 100)
untrained_models = [KNeighborsClassifier(n_neighbors=k) for k in neighbor_ks]

k_fold_scores = [cross_val_score(estimator=m, X=df_scaled[features], y=df_scaled['class'], cv=10) for m in untrained_models]

plt.plot(k_fold_scores)
plt.show()

mean_xv_scores = [s.mean() for s in k_fold_scores]
plt.errorbar(neighbor_ks, mean_xv_scores, yerr=[s.std() for s in k_fold_scores])
plt.show()

knn_best_k_xv = np.asarray(mean_xv_scores).argmax()
knn_best_k_train = np.asarray(train_scores).argmax()
knn_best_k_test = np.asarray(test_scores).argmax()
print(ks[knn_best_k_xv], ks[knn_best_k_train], ks[knn_best_k_test])
plt.semilogx(ks, train_scores, ks, test_scores, ks, mean_xv_scores)
plt.legend(["Train", "Test", "XV"])

knn_best_model_xv = models[knn_best_k_xv]

n_bootstrap_samples = 1000
bootstrap_test_score = np.zeros(n_bootstrap_samples)

for i in range(n_bootstrap_samples):
    bootstrap_test_sample = sklearn.utils.resample(test_data_s, replace=True, n_samples=N_test)
    bootstrap_test_score[i] = accuracy_score(bootstrap_test_sample['class'], knn_best_model_xv.predict(bootstrap_test_sample[features]))

plt.hist(bootstrap_test_score)
plt.title('Bootstrapped test scores for best kNN model')
plt.show()

