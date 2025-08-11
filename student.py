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
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tkinter import messagebox as msg
from django.db.models.expressions import result
from parso.python.tree import String
from scipy.stats import alpha

def KNN_student_model(STG,SCG,STR,LPR,PEG) -> int :
    # 'STG', 'SCG', 'STR', 'LPR', 'PEG', ' UNS'
    df= pd.read_csv ('studentknowledge.csv')
    # print(df.head(5).to_string())
    # print(df.dtypes)
    # print(df.shape)
    print(df.columns)
    # print(df.info)

    print(df.describe())
    # print(df.isna().sum())
    # print(df.groupby(' UNS')['SCG'].count())
    # # print(df[' UNS'].unique())

    #-----------------------------------------------mapping-------------------------------------------------------------
    # mapping
    df[' UNS'] = df[' UNS'].map({
        "High": 0,
        "Low": 1,
        "Middle": 2,
        "Very Low": 3,
        "very_low": 4
    })
    #
    # ---------------------------------------------vertical cut---------------------------------------------
    # verical cut
    X=df.drop(' UNS',axis=1)
    # print(X)
    Y=df[' UNS']
    # print(Y)

    # --------------------------------------horizontal cut-------------------------------------------
    # horizontal cut : sequence should be correct : X_train, X_test, Y_train, Y_test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)
    # ---------------------------------------------------------------------------------


    # print(df.describe())


    # --------------------------------------KNN model fit-------------------------------------------

    Knn_new=KNeighborsClassifier(n_neighbors=5)
    Knn_new.fit(X_train,Y_train)
    y_predict=Knn_new.predict(X_test)

    print(y_predict)

    # --------------------------------------accuracy model-------------------------------------------
    print(accuracy_score(Y_test,y_predict))
    print(classification_report(Y_test, y_predict))
    print(confusion_matrix(Y_test, y_predict))

    # --------------------------------------elbow method----------------------------------------------

    # new_result=[]
    #
    # for i in range (1,31):
    #     Knn_new = KNeighborsClassifier(n_neighbors=i)
    #     Knn_new.fit(X_train, Y_train)
    #     y_predict = Knn_new.predict(X_test)
    #     test_error=1-accuracy_score(Y_test, y_predict)
    #     new_result.append(test_error)
    #
    # print(new_result)
    #
    # sns.lineplot(new_result)
    # plt.show()
    x_params = [[STG,SCG,STR,LPR,PEG]]
    result = Knn_new.predict_proba(x_params)
    return result[0]


from tkinter import Tk, ttk, Label, Entry, StringVar, Button

knn_form = Tk()
knn_form.title('KNN')
knn_form.geometry('400x250')

def student_predict():
    STG = float(txtSTG.get())
    SCG = float(txtSCG.get())
    STR = float(txtSTR.get())
    LPR = float(txtLPR.get())
    PEG = float(txtPEG.get())
    probs = KNN_student_model(STG, SCG, STR, LPR, PEG)
    predicted_class_index = np.argmax(probs)

    label_map = {

        0: "High",
        1: "Low",
        2: "Middle",
        3: "Very Low",
        4: "very_low"
    }

    predicted_label = label_map[predicted_class_index]
    msg.showinfo('KNN Result', f'Predicted Label: {predicted_label}\nProbabilities: {probs}')


# STG,SCG,STR,LPR,PEG
lbl_STG = Label(knn_form,text='STG: ')
lbl_STG.grid(column=0, row=0, padx=10, pady=10)

txtSTG = StringVar()
ent_STG = Entry(knn_form, width=40, textvariable=txtSTG)
ent_STG.grid(column=1, row=0, padx=10, pady=10)

lbl_SCG = Label(knn_form,text='SCG: ')
lbl_SCG.grid(column=0, row=1, padx=10, pady=10)

txtSCG = StringVar()
ent_SCG = Entry(knn_form, width=40, textvariable=txtSCG)
ent_SCG.grid(column=1, row=1, padx=10, pady=10)


lbl_STR = Label(knn_form,text='STR: ')
lbl_STR.grid(column=0, row=2, padx=10, pady=10)

txtSTR = StringVar()
ent_STR = Entry(knn_form, width=40, textvariable=txtSTR)
ent_STR.grid(column=1, row=2, padx=10, pady=10)

lbl_LPR = Label(knn_form,text='LPR: ')
lbl_LPR.grid(column=0, row=3, padx=10, pady=10)

txtLPR = StringVar()
ent_LPR = Entry(knn_form, width=40, textvariable=txtLPR)
ent_LPR.grid(column=1, row=3, padx=10, pady=10)

lbl_PEG = Label(knn_form,text='PEG: ')
lbl_PEG.grid(column=0, row=4, padx=10, pady=10)

txtPEG= StringVar()
ent_PEG = Entry(knn_form, width=40, textvariable=txtPEG)
ent_PEG.grid(column=1, row=4, padx=10, pady=10)



btn_student_predict = Button(knn_form, width=20,text='Predict KNN Result', command=student_predict )
btn_student_predict.grid(column=1, row=5, padx=10, pady=10, sticky='N')

knn_form.mainloop()




# 0.08,0.08,0.1,0.24,0.9



