import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
emzirme_pdfile = pd.read_excel('ML/emzirme082023_v4.xlsx')

# Target Variable (y)
y = emzirme_pdfile['taburculuktakiKilogram']

# Create Predictive Features Array (X) Consisting of Columns
predictionFeatures = ['dogumAgirligiGram', 'kilo1.gun', 'kilo2.gun', 'kilo3.gun', 'cinsiyeti', 'gebelikHaftasi', 
                      'gebelikHaftaGunu', 'takipteKacGun', '1.GunTotali', 'aldigiMamaMiktari1.Gun', 'beslenmeTotali2.Gun',
                      'beslenmeMamaMiktari2.GunCC', 'beslenme2.GunAnneSutuCC', 'beslenmeTotali3.Gun', 'aldigiMamaMiktari3.Gun',
                      'aldigiAnneSutu3.Gun', 'beslenmeTotaliTaburculuk', 'taburculuktaMamaMiktari', 'aldigiAnneSutuTaburculuk',
                      'taburculuktaAnneSutu111', 'kacGunOGKullandi', 'postNatalGunEmzirme']
X = emzirme_pdfile[predictionFeatures] 

# Create Model
emzirmeModel = DecisionTreeRegressor(random_state=1)
emzirmeModel.fit(X, y)

# Predict
predictions = emzirmeModel.predict(X)
print("Showing Predictions: \n")
print(predictions)

# Evaluate
print(mean_absolute_error(y, predictions))

# Important Note:
# Since we haven't splited the data, our model cheats. MAE is very low.