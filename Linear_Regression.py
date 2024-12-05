######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import isnull

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.shape   #Out[6]: (200, 4)
df.head()

df["TV"]
X = df[["TV"]]      #bağımsız değişkenimiz
y = df[["sales"]]   #bağımlı değişken

##########################
# Model
##########################
df = pd.read_csv("datasets/advertising.csv")
df.shape   #Out[6]: (200, 4)
print(df.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).T)
X = df[["TV"]]      #bağımsız değişken
y = df[["sales"]]   #bağımlı değişken
reg_model = LinearRegression().fit(X, y)
# y_hat = b + w*TV
# sabit (b - bias)  #b yani bias için intercept
reg_model.intercept_[0]   #Out[16]: b: 7.032593549127687
# tv'nin katsayısı (w1)   #bazı kaybaklarda teta, coeff. ifadesi tercih edilir
reg_model.coef_[0][0]     #ağırlık w:Out[17]: 0.047536640433019806

########################################################################################################
# Tahmin
########################################################################################################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
# b + w*150   150 birimlik bir tv harcaması olursa bu durumda 14 birimlik satış harcaması olacaktır.
reg_model.intercept_[0] + reg_model.coef_[0][0]*150   #Out[22]: 14.163089614080658

# 280 birimlik tv harcaması olsa ne kadar satış olur?
# b + w*280    #
reg_model.intercept_[0] + reg_model.coef_[0][0]*280  #Out[20]: 20.342852870373232

df.describe().T
#            count   mean   std  min   25%    50%    75%    max
# TV        200.00 147.04 85.85 0.70 74.38 149.75 218.82 296.40 #tv nin max değeri 296 biz 500 girdik
# radio     200.00  23.26 14.85 0.00  9.97  22.90  36.52  49.60
# newspaper 200.00  30.55 21.78 0.30 12.75  25.75  45.10 114.00
# sales     200.00  14.02  5.22 1.60 10.38  12.90  17.40  27.00  #max değer zaten 27 biz 500 ü 30 tahmin ettik
#burada elimde örnek veride gözlenmemiş bir değer olsa bile ben bunu öğrendiğim modele sorabilirim

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},  #regplot metodunu kullandık x:bağımsız y:bağımlı değişken
                ci=False, color="r")  #ci:güven aralığı False yani ekleme dedik  regülasyon çizgisi rengi R

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")   #round(reg_model.intercept_[0], 2) virgülden sonra iki basamak al
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310) #-10 dan 310 a kadar x eksenini görselleştir
plt.ylim(bottom=0)  #y limite sıfırdan başla
plt.show()


########################################################################################################
# Tahmin Başarısı
########################################################################################################

# MSE  derkş bana gerçek (y) ve tahmin edilen değerleri (y_pred) ver

y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
#mean_squared_error(y, y_pred)
# Out[29]: 10.512652915656759

y.mean()
# sales   14.02  #gerçek değerlerin ortalaması

y.std()  # sales   5.22    yani   14+5   --   14-5  arasında değer alır
y.mean()
# RMSE  #root mean squared error   >>   MSE nin kareköküdür
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)  #farklı metriklerin hataları birbiri ile karşılaştırılmaz
# 2.54

# R-KARE
reg_model.score(X, y)   #Out[33]: 0.611875050850071
#bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir
#yani bu verisetinde bir değişken var, tv değişkeninin satış değişikliğindeki değişikliği
#açıklama yüzdesidir.
#yani bu modelde bağımsız değişkenler bağımlı değişkenin %61 ini açıklayabilmektedir.

############################################################################################################
# Multiple Linear Regression
############################################################################################################

df = pd.read_csv("datasets/advertising.csv")
df.head()
df.shape  #Out[46]: (200, 4)
#       TV  radio  newspaper  sales
# 0 230.10  37.80      69.20  22.10
# 1  44.50  39.30      45.10  10.40
# 2  17.20  45.90      69.30   9.30
# 3 151.50  41.30      58.50  18.50
# 4 180.80  10.80      58.40  12.90

########################################################################################################
# Model
########################################################################################################
X = df.drop('sales', axis=1)
y = df[["sales"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

y_test.shape    #Out[45]: (40, 1)    200 satırın %20 si 40 olacaktır. test kısmı
X_train.shape   #Out[44]: (160, 3)   200 satırın %80 i  160 olacaktır. train kısmı
y_train.shape   #Out[42]: (160, 1)   1 tane bağımlı değişkenimiz var

reg_model = LinearRegression().fit(X_train, y_train)
# sabit (b - bias)
reg_model.intercept_   #Out[51]: array([2.90794702])   [0]

# coefficients (w - weights)            TV           radio       newspaper
reg_model.coef_       #Out[54]: array([[0.0468431 , 0.17854434, 0.00258619]])   3 değişkenin ağırlığı

# Sales : TV: 30      b+ w1*30  =  2.90 + ( 0.046 * 30 ) + (0.178 * 10) + (0.0025* 40 ) = 5,88
# Sales : radio: 10
# Sales : newspaper: 40
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30], [10], [40]]
#    0    #transpozunu almazsan satırlarda indexler kalır
# 0  30
# 1  10
# 2  40
yeni_veri = pd.DataFrame(yeni_veri).T  #veriyi DATAFRAME e çevir
#     0   1   2
# 0  30  10  40

################################################################################
# Tahmin
################################################################################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?
df.head()
# coefficients (w - weights)            TV           radio       newspaper
reg_model.coef_       #Out[54]: array([[0.0468431 , 0.17854434, 0.00258619]])   3 değişkenin ağırlığı
reg_model.intercept_  # # 2.90

# Sales : TV: 30      b+ w1*30  =  2.90 + ( 0.046 * 30 ) + (0.178 * 10) + (0.0025* 40 ) = 5,88
# Sales : radio: 10
# Sales : newspaper: 40
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30], [10], [40]]
#    0    #transpozunu almazsan satırlarda indexler kalır
# 0  30
# 1  10
# 2  40
yeni_veri = pd.DataFrame(yeni_veri).T  #veriyi DATAFRAME e çevir
#     0   1   2
# 0  30  10  40

# reg_model = LinearRegression().fit(X_train, y_train)
reg_model.predict(yeni_veri)     #Out[65]: array([[6.202131]])  tahminleme

########################################################################################################
# Tahmin Başarısını Değerlendirme
########################################################################################################
#regülasyon modelini %80 ile train setinde kurduk,
#traing setinni bağımlı değişkenini de tahmin edip kenarda saklayabilir
#onun hata kareler ortalamasının karekökü değerine erişebiliriz
# Train RMSE

y_pred = reg_model.predict(X_train)
# #array([[ 3.65921577],
#        [ 7.25612637],
#        [ 6.00481636],
#        [18.46169785],
#        [ 8.37406584], .....   tahmin değerleri

# y_train   gerçek sales rakamları
# Out[70]:
#      sales
# 108   5.30
# 107   8.70
# 189   6.70

y_pred = reg_model.predict(X_train)
# Train RMSE   train root mean squared error hatamız
# modeli train seti üzerinden kurduk
np.sqrt(mean_squared_error(y_train, y_pred))  #modeli train seti üzerinden kurduk ve bunun hatasını değerlendirmek istiyorsak
# 1.73

# TRAIN RKARE  #bağımsız değişkenlerin bağımlı değişkeni etkileme açıklama oranı
reg_model.score(X_train, y_train)
#Out[71]: 0.8959372632325174 neredeyse %90 geldi
#####--------------------------------------------------------------------#####
# Test RMSE
y_pred = reg_model.predict(X_test)    #train üzerinden kurduğumuz modele, modelin görmediği test setini soruyoruz
#ben sana set göndereceğim sen bunu değerlendir. test setinin x lerini yani bağımsız değişkenlerini soruyoruz
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)  #Out[76]: 0.8927605914615387 önceki değere yakın


# 10 Katlı CV RMSE     ##CROSS VALIDATION
np.mean(np.sqrt(-cross_val_score(reg_model,   #REG MODELİ KULLANDIK  #sqrt karekökü, mena ortalaması  RMSE
                                 X,   #X VE Y'yi olduğu gibi yazdım, bütün veri üzerinden değerlendiriyorum.
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))   #Out[77]: 1.6913531708051792
#Cross Val. Score Negative 'yi' yeni negatif ort. hatayı veriyor, bu sebeple bunu eksi ile çaarptık
#array([-3.56038438, -3.29767522, -2.08943356, -2.82474283, -1.3027754 ,
       # -1.74163618, -8.17338214, -2.11409746, -3.04273109, -2.45281793])
# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

