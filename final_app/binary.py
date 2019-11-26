import joblib
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('data.csv', header=None)
df.rename(columns={0: 'User ID', 1: 'Longeivity',2:'Length of screen name', 3:'Does the profile have a description', 4 : 'Length of the description', 5:'Does the profile have a URL',6:'Followee count of the user',7:'Follower count of the user', 8:'Followee-by-follower ratio', 9:'Klout Score', 10:'Total number of tweets',
11:'Number of direct mentions per tweet', 12:'Number of URLs per tweet', 13:'Number of hashtags per tweet', 14:'Number of tweets per day', 15:'Number of re-tweets per day', 16:'Number of re-tweets per tweet',17:'Botometer score',18:'Tweeting likelihood for Monday',19:'Tweeting likelihood for Tuesday',20:'Tweeting likelihood for Wednesday',
21:'Tweeting likelihood for Thursday',22:'Tweeting likelihood for Friday',23:'Tweeting likelihood for Saturday',24:'Tweeting likelihood for Sunday',25:'Re-tweeting likelihood for Monday',26:'Re-tweeting likelihood for Tuesday',27:'Re-tweeting likelihood for Wednesday',28:'Re-tweeting likelihood for Thursday',29:'Re-tweeting likelihood for Friday',
30:'Re-tweeting likelihood for Saturday',31:'Re-tweeting likelihood for Sunday',32:'Regularity of tweeting activity for Monday',33:'Regularity of tweeting activity for Tuesday',34:'Regularity of tweeting activity for Wednesday',35:'Regularity of tweeting activity for Thursday',36:'Regularity of tweeting activity for Friday',
37:'Regularity of tweeting activity for Saturday',38:'Regularity of tweeting activity for Sunday',39:'Regularity of re-tweeting activity for Monday',40:'Regularity of re-tweeting activity for Tuesday',41:'Regularity of re-tweeting activity for Wednesday',42:'Regularity of re-tweeting activity for Thursday',43:'Regularity of re-tweeting activity for Friday',
44:'Regularity of re-tweeting activity for Saturday',45:'Regularity of re-tweeting activity for Sunday',46:'Tweet steadiness',47:'Re-tweet steadiness',48:'Maximum tweet likelihood for Monday',49:'Maximum tweet likelihood for Tuesday',50:'Maximum tweet likelihood for Wednesday',51:'Maximum tweet likelihood for Thursday',52:'Maximum tweet likelihood for Friday',
53:'Maximum tweet likelihood for Saturday',54:'Maximum tweet likelihood for Sunday',55:'Maximum re-tweet likelihood for Monday',56:'Maximum re-tweet likelihood for Tuesday',57:'Maximum re-tweet likelihood for Wednesday',58:'Maximum re-tweet likelihood for Thursday',59:'Maximum re-tweet likelihood for Friday',60:'Maximum re-tweet likelihood for Saturday',
61:'Maximum re-tweet likelihood for Sunday',62:'Standard deviation of retweet counts for all user-generated tweets',63:'Mean of log-time difference between consecutive re-tweets',64:'Standard deviation of log-time difference between consecutive re-tweets',65:'Annotation (0: Bot, 1: Normal customers, 2: Promotional customers, 3: Genuine users)'}, inplace=True)
df.to_csv('data3.csv', index=False) # save to new csv file
fin = open("data3.csv")

data_points = []
labels = []
macro_stats_lr = []
macro_stats_svm = []
macro_stats_lr_w = []
macro_stats_svm_w = []

num_lines = 0

for line in fin:

	if num_lines!=0:
		toks = line.strip().split(",")
		tmp = toks[1:-1]#remove last two column
		print(tmp)
		inp = [float(tok) for tok in tmp]
		label = int(toks[-1])
		data_points.append(inp)
		labels.append(label)
	num_lines +=1

y = labels #dependent
#print(data_points)
#print(labels)
for i in range(len(y)):

		if y[i] == 0 or y[i] == 1 or y[i] == 2:

			y[i] = 0

		elif y[i] == 3:

			y[i] = 1          #genuine user

class_names = ['genuine', 'non-genuine']
data_points = np.array(data_points)
y = np.array(y)
kf = StratifiedKFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(data_points, y):

	X_train, X_test = data_points[train_index], data_points[test_index]
	y_train, y_test = y[train_index], y[test_index]


	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)

	#clf_lr = LogisticRegression()
	#clf_lr.fit(X_train, y_train)
	#y_pred_lr = clf_lr.predict(X_test)
	#classification_report_lr = classification_report(y_test, y_pred_lr, target_names=class_names)
	#macro_stats_lr.append(precision_recall_fscore_support(y_test, y_pred_lr, average='macro'))
	#macro_stats_lr_w.append(precision_recall_fscore_support(y_test, y_pred_lr, average='weighted'))
	#print(classification_report_lr)
	clf = svm.SVC()
	clf.fit(X_train, y_train)
	y_out = clf.predict(X_test)
	clf_accuracy = clf.score(X_test, y_test)
	classifiy_report_svm = classification_report(y_test, y_out, target_names=class_names)
	print(classifiy_report_svm)
	macro_stats_svm.append(precision_recall_fscore_support(y_test, y_out, average='macro'))
	macro_stats_svm_w.append(precision_recall_fscore_support(y_test, y_out, average='macro'))
#print("LR MACRO AVG SCORE")
#print(macro_stats_lr)
#print("LR WEIGHTED AVG  SCORE")
#print(macro_stats_lr_w)
#print("SVM MACRO AVG  SCORE")
#print(macro_stats_svm)
#print("SVM WEIGHTED AVG  SCORE")
#print(macro_stats_svm_w)
print("Macro AVG (Precision, recall, f1-score, support)")
print("Precision: ", float(sum([p[0] for p in macro_stats_svm]))/len(macro_stats_svm))
print("Recall: ", float(sum([p[1] for p in macro_stats_svm]))/len(macro_stats_svm))
print("F1-score: ", float(sum([p[2] for p in macro_stats_svm]))/len(macro_stats_svm))
print("\n")
print("Macro WEIGHTED(Precision, recall, f1-score, support)")
print("Precision: ", float(sum([p[0] for p in macro_stats_svm_w]))/len(macro_stats_svm_w))
print("Recall: ", float(sum([p[1] for p in macro_stats_svm_w]))/len(macro_stats_svm_w))
print("F1-score: ", float(sum([p[2] for p in macro_stats_svm_w]))/len(macro_stats_svm_w))
from sklearn.externals import joblib
joblib.dump(clf, 'model3.pkl')
print("Model dumped!")
print(y_out)
