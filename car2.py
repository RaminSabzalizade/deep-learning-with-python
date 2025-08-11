import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tkinter import Tk, Label, Button, StringVar, ttk, messagebox as msg

# ------------------ Load & map dataset ------------------
columns = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv('car.data', names=columns, header=None)

map_3 = {'vhigh':3,'high':2,'med':1,'low':0}
df['buying'] = df['buying'].map(map_3)
df['maint']  = df['maint'].map(map_3)
df['safety'] = df['safety'].map(map_3)

lug_map = {'big':2,'med':1,'small':0}
df['lug_boot'] = df['lug_boot'].map(lug_map)

class_map = {'vgood':3,'good':2,'acc':1,'unacc':0}
class_rev = {v:k for k,v in class_map.items()}
df['class'] = df['class'].map(class_map)

df['doors'] = df['doors'].map({'5more':5}).fillna(df['doors']).astype(int)
df['persons'] = df['persons'].map({'more':5}).fillna(df['persons']).astype(int)

X = df.drop('class', axis=1)
y = df['class']

# ------------------ Train/test + pipeline ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
print("Validation accuracy:", acc)

# (Optional) retrain on full data for best final model
pipe.fit(X, y)

# ------------------ GUI ------------------
root = Tk()
root.title("Car Evaluation (KNN)")

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
    Label(root, text=label).grid(row=r, column=0, padx=8, pady=6, sticky='e')
    cb = ttk.Combobox(root, textvariable=var, values=values, state='readonly', width=10)
    cb.grid(row=r, column=1, padx=8, pady=6, sticky='w')
    return cb

add_combo('buying:',  buying_var,  buying_vals,  0)
add_combo('maint:',   maint_var,   maint_vals,   1)
add_combo('doors:',   doors_var,   doors_vals,   2)
add_combo('persons:', persons_var, persons_vals, 3)
add_combo('lug_boot:',lug_var,     lug_vals,     4)
add_combo('safety:',  safety_var,  safety_vals,  5)

result_label = Label(root, text=f"Validation accuracy: {acc:.3f}")
result_label.grid(row=6, column=0, columnspan=2, padx=8, pady=10)

def predict():
    # Map GUI selections to numeric features (same mapping as training)
    rec = {
        'buying':  map_3[buying_var.get()],
        'maint':   map_3[maint_var.get()],
        'doors':   5 if doors_var.get() == '5more' else int(doors_var.get()),
        'persons': 5 if persons_var.get() == 'more' else int(persons_var.get()),
        'lug_boot': lug_map[lug_var.get()],
        'safety':  map_3[safety_var.get()],
    }
    X_new = pd.DataFrame([rec])
    pred = pipe.predict(X_new)[0]
    msg.showinfo("Prediction", f"Predicted class: {class_rev[pred]}")

Button(root, text="Predict", command=predict).grid(row=7, column=0, columnspan=2, pady=12)

root.mainloop()
