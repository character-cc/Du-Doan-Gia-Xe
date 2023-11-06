import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import font
from sklearn.datasets import load_iris
import joblib
from sklearn.preprocessing import StandardScaler
loaded_model = joblib.load('random.pkl')
#data = pd.read_csv('data_with2.csv')
import math
import numpy as np
#dât = pd.DataFrame({col: [0] for col in data.columns})
df = pd.read_csv('car.csv')
x = df.drop('Price',axis=1)
#y = df['Price']
scaler = StandardScaler()
scaler.fit(x[['Mileage','EngineV']])
inputs_scaled = scaler.transform(x[['Mileage','EngineV']])
scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])
x =scaled_data.join(x.drop(['Mileage','EngineV'],axis=1))
y = loaded_model.predict(x)
test = df['Price']
differences = []
dem = 0
for i in range(len(test)):
    diff = test[i] - y[i]
    print(test[i] , y[i])
    differences.append(diff)
    if(diff < 3000):
        dem = dem + 1


# In sự chênh lệch giữa từng cặp giá trị
for i, diff in enumerate(differences):
    print(f"Sự chênh lệch giữa giá trị thực tế và giá trị dự đoán tại index {i}: {diff}")
print(dem)
count = len(df[df['Price'] > 10000])

print("Số lượng giá trị lớn hơn 20000 trong cột 'Price':", count)
#print(  new_data.iloc[0].values.reshape(1, -1))
print(x.iloc[0])
def predict():
   # try:
        new_data = pd.DataFrame({col: [0] for col in df.columns})
        new_data = new_data.drop('Price', axis=1)
        mi = float(Mileage.get())
        en = float(EngineV.get())
        br = (selected_brand.get())
        et = (selected_Engine_Type.get())
        bo = (selected_body.get())
        re = Registration.get()
        input_data = [mi, en, br, et,bo,re]
        new_data.at[0, 'Mileage'] = mi
        new_data.at[0, 'EngineV'] = en
        new_data.at[0, br] = True
        if(et != "Engine Type_Diesel") :
           new_data.at[0, et] = True
        new_data.at[0, bo] = True              
        if re == "Yes" :
          new_data.at[0, 'Registration_yes'] = True
        x = new_data
        #print(x.iloc[0])
        inputs_scaled = scaler.transform(x[['Mileage','EngineV']])
        scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])
        x =scaled_data.join(x.drop(['Mileage','EngineV'],axis=1)) 
        print(x.iloc[0])
        y_pred  = loaded_model.predict(x)
        #print(np.exp(8.9))
        result_label.config(text=f"Giá xe dự đoán: {(y_pred)}")
#    except ValueError:
  #      messagebox.showerror("Lỗi", "Vui lòng nhập giá trị số cho các thông số.")
app = Tk()
app.title("Dự Đoán Giá Xe")

# Tạo một phông chữ với kích thước lớn
custom_font = font.nametofont("TkDefaultFont")
custom_font.configure(size=16)  # Đặt kích thước chữ là 16

label_Mileage = Label(app, text= "Mileage")
label_Mileage.pack()

Mileage = tk.Entry(app, font=custom_font)
Mileage.pack()

label_EngineV = Label(app, text= "EngineV")
label_EngineV.pack()

EngineV = tk.Entry(app, font=custom_font)
EngineV.pack()
label_brand = tk.Label(app, text="Chọn Brand")
label_brand.pack()
brand = ["Brand_BMW", "Brand_Mercedes-Benz", "Brand_Mitsubishi", "Brand_Renault",
           "Brand_Toyota","Brand_Volkswagen"]
selected_brand = tk.StringVar()
# Tạo trường chọn và hiển thị nó tự động
dropdown_brand = tk.OptionMenu(app, selected_brand, *brand)
dropdown_brand.pack()

selected_brand.set(brand[0])  

label_body = tk.Label(app, text="Chọn Body")
label_body.pack()
body = ["Body_hatch", "Body_sedan", "Body_vagon", "Body_van",
           "Body_other"]
selected_body = tk.StringVar()
# Tạo trường chọn và hiển thị nó tự động
dropdown_body = tk.OptionMenu(app, selected_body, *body)
dropdown_body.pack()

selected_body.set(body[0])  

label_Engine_Type = tk.Label(app, text="Chọn Engine Type")
label_Engine_Type.pack()
Engine_Type = ["Engine Type_Gas", "Engine Type_Petrol", "Engine Type_Other","Engine Type_Diesel"]
selected_Engine_Type = tk.StringVar()
# Tạo trường chọn và hiển thị nó tự động
dropdown_Engine_Type = tk.OptionMenu(app, selected_Engine_Type, *Engine_Type)
dropdown_Engine_Type.pack()

label_Registration = tk.Label(app, text="Registration")
label_Registration.pack()
selected_Engine_Type.set(Engine_Type[0])  
# Biến để theo dõi lựa chọn của người dùng
def on_radio_button_selected():
    Registration.set(Registration.get())

Registration = tk.StringVar()

# Tạo Frame để đặt radio buttons vào


# Tạo các radio buttons
radio_button_yes = tk.Radiobutton(app, text="Yes", variable=Registration, value="Yes", command=on_radio_button_selected)
radio_button_no = tk.Radiobutton(app, text="No", variable=Registration, value="No", command=on_radio_button_selected)

# Đặt giá trị mặc định cho radio button
Registration.set("Yes")

# Hiển thị radio buttons ở giữa
radio_button_yes.pack()
radio_button_no.pack()

predict_button = Button(app, text="Dự Đoán", command=predict)
predict_button.pack()

result_label = Label(app,text="")
result_label.pack()

app.mainloop()