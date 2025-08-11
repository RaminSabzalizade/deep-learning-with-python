import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as ms
from openpyxl.styles.alignment import horizontal_alignments
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import numpy as np
from tkinter import messagebox as msg
from django.db.models.expressions import result
from parso.python.tree import String
from scipy.stats import alpha
from tkinter import Tk, ttk, Label, Entry, StringVar, Button

def localization(wifi1,wifi2,wifi3,wifi4,wifi5,wifi6,wifi7)-> int :

    column_names=['wifi1','wifi2','wifi3','wifi4','wifi5','wifi6','wifi7','room']
    df= pd.read_csv('wifi_localization.txt',sep='\t',header=None,names=column_names)

    # print(df.head(5).to_string())
    # print(df.dtypes)
    # print(df.shape)
    # print(df.columns)
    # print(df.info)
    # print(df.describe())
    # print(df.isna().sum())
    # # ----------------------------------cut vertical  -------------------------------------
    X=df.drop("room",axis=1)
    Y=df['room']
    print(Y)
    # # ----------------------------------mms-------------------------------------
    mms=MinMaxScaler()
    X=mms.fit_transform(X)

    #  ----------------------------------cut Horizontal -------------------------------------
    X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=42,shuffle=True)


    #  ----------------------------------model train and test -------------------------------------
    KNN_model=KNeighborsClassifier(n_neighbors=3)
    KNN_model.fit(X_train,Y_train)
    y_predict=KNN_model.predict(X_test)


    print(y_predict)
    print(accuracy_score(Y_test, y_predict))
    print(classification_report(Y_test, y_predict))
    print(confusion_matrix(Y_test, y_predict))
    #  ----------------------------------several neighbors  -------------------------------------
    # higher neighbor increase the error

    # result_value_localization=[]
    # for i in range (1,20):
    #     KNN_model = KNeighborsClassifier(n_neighbors=i)
    #     KNN_model.fit(X_train, Y_train)
    #     y_predict = KNN_model.predict(X_test)
    #     result_value_localization.append(accuracy_score(Y_test, y_predict))
    # print(result_value_localization)
    # sns.lineplot(result_value_localization)
    # plt.show()
    x_param=[[wifi1,wifi2,wifi3,wifi4,wifi5,wifi6,wifi7]]
    x_param_scaled = mms.transform(x_param)
    probs = KNN_model.predict_proba(x_param_scaled)[0]
    classes = KNN_model.classes_
    return probs,classes


def localization_predict():
    wifi1 = float(txtwifi1.get())
    wifi2 = float(txtwifi2.get())
    wifi3 = float(txtwifi3.get())
    wifi4 = float(txtwifi4.get())
    wifi5 = float(txtwifi5.get())
    wifi6 = float(txtwifi6.get())
    wifi7 = float(txtwifi7.get())
    probs, classes = localization(wifi1,wifi2,wifi3,wifi4,wifi5,wifi6,wifi7)
    best_idx = int(np.argmax(probs))
    best_room = classes[best_idx]
    lines = "\n".join([f"Room {c}: {p:.2f}" for c, p in zip(classes, probs)])

    msg.showinfo('KNN Result', f'Predicted room: {best_room}\n\nProbabilities:\n{lines}')




knn_localization_form = Tk()
knn_localization_form.title('KNN_Localization')
knn_localization_form.geometry('400x350')


lbl_wifi1 = Label(knn_localization_form,text='wifi1: ')
lbl_wifi1.grid(column=0, row=0, padx=10, pady=10)

txtwifi1 = StringVar()
ent_wifi1 = Entry(knn_localization_form, width=40, textvariable=txtwifi1)
ent_wifi1.grid(column=1, row=0, padx=10, pady=10)
# --------------------------------------------------------------------
lbl_wifi2 = Label(knn_localization_form,text='wifi2: ')
lbl_wifi2.grid(column=0, row=1, padx=10, pady=10)

txtwifi2 = StringVar()
ent_wifi2 = Entry(knn_localization_form, width=40, textvariable=txtwifi2)
ent_wifi2.grid(column=1, row=1, padx=10, pady=10)

# --------------------------------------------------------------------

lbl_wifi3 = Label(knn_localization_form,text='wifi3: ')
lbl_wifi3.grid(column=0, row=2, padx=10, pady=10)

txtwifi3 = StringVar()
ent_wifi3 = Entry(knn_localization_form, width=40, textvariable=txtwifi3)
ent_wifi3.grid(column=1, row=2, padx=10, pady=10)
# --------------------------------------------------------------------

lbl_wifi4 = Label(knn_localization_form,text='wifi4: ')
lbl_wifi4.grid(column=0, row=3, padx=10, pady=10)

txtwifi4 = StringVar()
ent_wifi4 = Entry(knn_localization_form, width=40, textvariable=txtwifi4)
ent_wifi4.grid(column=1, row=3, padx=10, pady=10)
# --------------------------------------------------------------------

lbl_wifi5 = Label(knn_localization_form,text='wifi5: ')
lbl_wifi5.grid(column=0, row=4, padx=10, pady=10)


txtwifi5 = StringVar()
ent_wifi5 = Entry(knn_localization_form, width=40, textvariable=txtwifi5)
ent_wifi5.grid(column=1, row=4, padx=10, pady=10)
# --------------------------------------------------------------------

lbl_wifi6 = Label(knn_localization_form,text='wifi6: ')
lbl_wifi6.grid(column=0, row=5, padx=10, pady=10)


txtwifi6 = StringVar()
ent_wifi6 = Entry(knn_localization_form, width=40, textvariable=txtwifi6)
ent_wifi6.grid(column=1, row=5, padx=10, pady=10)
# --------------------------------------------------------------------

lbl_wifi7 = Label(knn_localization_form,text='wifi7: ')
lbl_wifi7.grid(column=0, row=6, padx=10, pady=10)


txtwifi7= StringVar()
ent_wifi7 = Entry(knn_localization_form, width=40, textvariable=txtwifi7)
ent_wifi7.grid(column=1, row=6, padx=10, pady=10)
# --------------------------------------------------------------------



btn_student_predict = Button(knn_localization_form, width=20,text='Predict KNN Result', command=localization_predict )
btn_student_predict.grid(column=1, row=8, padx=10, pady=10, sticky='N')

knn_localization_form.mainloop()














