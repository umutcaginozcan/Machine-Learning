import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

emzirme_data = pd.read_excel('ML/emzirme082023_v4.xlsx')
emzirme_data.dropna(inplace=True) # This single line of code causes 956 rows of data to be lost, and ultimately our model to be rubbish.

predictionFeatures = ['dogumAgirligiGram', 'kilo1.gun', 'kilo2.gun', 'kilo3.gun', 'cinsiyeti', 'gebelikHaftasi', 
                      'gebelikHaftaGunu', 'takipteKacGun', '1.GunTotali', 'aldigiMamaMiktari1.Gun', 'beslenmeTotali2.Gun',
                      'beslenmeMamaMiktari2.GunCC', 'beslenme2.GunAnneSutuCC', 'beslenmeTotali3.Gun', 'aldigiMamaMiktari3.Gun',
                      'aldigiAnneSutu3.Gun', 'beslenmeTotaliTaburculuk', 'taburculuktaMamaMiktari', 'aldigiAnneSutuTaburculuk',
                      'taburculuktaAnneSutu111', 'kacGunOGKullandi', 'postNatalGunEmzirme']
X = emzirme_data[predictionFeatures] 

# target variable 
y = emzirme_data['taburculuktakiKilogram']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

emzirme_model = DecisionTreeRegressor(random_state=1)
emzirme_model.fit(train_X, train_y)

# Predict
predictions = emzirme_model.predict(val_X)
print("Showing Predictions: \n")
print(predictions)

# Evaluate
print(mean_absolute_error(val_y, predictions))