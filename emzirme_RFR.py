import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
emzirme_pdfile = pd.read_excel('ML_Emzirme/emzirme082023_v4.xlsx')
print(emzirme_pdfile)

# Target Variable (y)
y = emzirme_pdfile['taburculuktakiKilogram']

# Create Predictive Features Array (X) Consisting of Columns
predictionFeatures = ['dogumAgirligiGram', 'kilo1.gun', 'kilo2.gun', 'kilo3.gun', 'cinsiyeti', 'gebelikHaftasi', 
                      'gebelikHaftaGunu', 'takipteKacGun', '1.GunTotali', 'aldigiMamaMiktari1.Gun', 'beslenmeTotali2.Gun',
                      'beslenmeMamaMiktari2.GunCC', 'beslenme2.GunAnneSutuCC', 'beslenmeTotali3.Gun', 'aldigiMamaMiktari3.Gun',
                      'aldigiAnneSutu3.Gun', 'beslenmeTotaliTaburculuk', 'taburculuktaMamaMiktari', 'aldigiAnneSutuTaburculuk',
                      'taburculuktaAnneSutu111', 'kacGunOGKullandi', 'postNatalGunEmzirme', 'sutDestegiVarsaKacOlcek', 'varsaTaburculuktaKacOlcek']
X = emzirme_pdfile[predictionFeatures] 

print("Data All: ")
dataAll = emzirme_pdfile['taburculuktakiKilogram'].describe()
print(dataAll)

# Split data into 2.
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.76, random_state=0)

# Create Model
emzirmeModel = DecisionTreeRegressor(max_leaf_nodes=400 ,random_state=1)
emzirmeModel.fit(train_X, train_y)
emzirmeForestModel = RandomForestRegressor()
emzirmeForestModel.fit(train_X, train_y)

# Predict
predictions = emzirmeModel.predict(val_X)
print("Showing Predictions: \n")
print(predictions)

print("\n")
predictions2 = emzirmeForestModel.predict(val_X)
print("Showing Predictions: \n")
print(predictions)

# Evaluate
print(mean_absolute_error(val_y, predictions))
print(mean_absolute_error(val_y, predictions2))