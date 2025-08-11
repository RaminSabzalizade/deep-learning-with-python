import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from tkinter import messagebox as msg
from tkinter import Tk, ttk, Label, Entry, StringVar, Button
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

def car_prediction(buying,maint,doors,persons,lug_boot,safety)  :
    # buying,maint,doors,persons,lug_boot,safety,class
    columns=['buying','maint','doors','persons','lug_boot','safety','class']
    df= pd.read_csv('car.data',sep=',',usecols=None,names=columns)
    print(df.columns)
    # print(df.isna().sum())
    # print(df.dtypes)
    # print(df.info())
    # print(df.head().to_string())
    # print(df.groupby('doors')['persons'].count())

    # -----------------------------------------Mapping---------------------------------------------------
    # v-high, high, med, low
    df['buying']=df['buying'].map({
                        'vhigh':3,
                        'high':2,
                        'med':1,
                        'low':0
    })
    df['maint']=df['maint'].map({
                        'vhigh':3,
                        'high':2,
                        'med':1,
                        'low':0
    })
    df['safety']=df['safety'].map({
                        'vhigh':3,
                        'high':2,
                        'med':1,
                        'low':0
    })
    df['lug_boot']=df['lug_boot'].map({
                        'big':2,
                        'med':1,
                        'small':0
    })
    df['class']=df['class'].map({
                        'vgood':3,
                        'good':2,
                        'acc':1,
                        'unacc':0
    })

    df['doors']=df['doors'].map({
                        '5more':5
    }).fillna(df['doors']).astype(int)

    df['persons']=df['persons'].map({
                    'more':5
    }).fillna(df['persons']).astype(int)

    # print(df['buying'].head().to_string())
    # print(df['buying'].unique())
    # print(df['safety'].unique())
    # print(df['maint'].unique())
    # print(df['class'].unique())
    # print(df['doors'].unique())
    # print(df['persons'].unique())
    # print(df['class'].unique())
    # print(df.info())
    # print(df.head().to_string())

    # -------------------------------------------cut-------------------------------------------------

    print(df)
    X=df.drop('class',axis=1)
    Y=df['class']
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=42,shuffle=True)
    # -------------------------------------------mms-------------------------------------------------
    mms=MinMaxScaler()
    X_train=mms.fit_transform(X_train)
    X_test=mms.transform(X_test)


    # print(df)
    # -------------------------------------------KNN-------------------------------------------------
    KNN_car= KNeighborsClassifier(n_neighbors=5)
    KNN_car.fit(X_train,Y_train)
    y_predict= KNN_car.predict(X_test)

    acc=accuracy_score(Y_test,y_predict,)
    a=confusion_matrix(Y_test,y_predict)
    print(classification_report(Y_test,y_predict,))
    # encode GUI inputs exactly like the dataframe
    map_v = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
    map_lug = {'big': 2, 'med': 1, 'small': 0}

    doors_num = 5 if doors == '5more' else int(doors)
    persons_num = 5 if persons == 'more' else int(persons)

    x_param_num = [[
        map_v[buying],
        map_v[maint],
        doors_num,
        persons_num,
        map_lug[lug_boot],
        map_v[safety]
    ]]
    x_param_scaled = mms.transform(x_param_num)
    probs = KNN_car.predict_proba(x_param_scaled)[0]
    classes = KNN_car.classes_
    return probs, classes


car_prediction_form = Tk()
car_prediction_form.title('car_prediction_form')
car_prediction_form.geometry('400x350')

# --- good output handler (define BEFORE button) ---
def car1_predict():
    probs, classes = car_prediction(
        buying_var.get(), maint_var.get(), doors_var.get(),
        persons_var.get(), lug_var.get(), safety_var.get()
    )
    inv_cls = {0:'unacc', 1:'acc', 2:'good', 3:'vgood'}

    best_idx   = int(np.argmax(probs))
    best_label = inv_cls[int(classes[best_idx])]

    lines = "\n".join([f"{inv_cls[int(c)]}: {p:.3f}" for c, p in zip(classes, probs)])
    msg.showinfo('KNN Result', f'Predicted class: {best_label}\n\nProbabilities:\n{lines}')

# Combobox options using original text categories
buying_vals  = ['vhigh','high','med','low']
maint_vals   = ['vhigh','high','med','low']
doors_vals   = ['2','3','4','5more']
persons_vals = ['2','4','more']
lug_vals     = ['small','med','big']
safety_vals  = ['vhigh','high','med','low']

# Tk variables
buying_var  = StringVar(value='med')
maint_var   = StringVar(value='med')
doors_var   = StringVar(value='4')
persons_var = StringVar(value='4')
lug_var     = StringVar(value='med')
safety_var  = StringVar(value='high')

# Row helper
def add_combo(label, var, values, r):
    Label(car_prediction_form, text=label).grid(row=r, column=0, padx=8, pady=6, sticky='e')
    cb = ttk.Combobox(car_prediction_form, textvariable=var, values=values, state='readonly', width=10)
    cb.grid(row=r, column=1, padx=8, pady=6, sticky='w')
    return cb

add_combo('buying:',  buying_var,  buying_vals,  0)
add_combo('maint:',   maint_var,   maint_vals,   1)
add_combo('doors:',   doors_var,   doors_vals,   2)
add_combo('persons:', persons_var, persons_vals, 3)
add_combo('lug_boot:',lug_var,     lug_vals,     4)
add_combo('safety:',  safety_var,  safety_vals,  5)

# --- create button THEN grid it ---
btn_car_predict = Button(
    car_prediction_form, width=20, text='Predict KNN Result',
    command=car1_predict
)
btn_car_predict.grid(column=0, row=8, padx=10, pady=10, sticky='N')

car_prediction_form.mainloop()


