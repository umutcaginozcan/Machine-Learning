# Define, Fit, Predict, Evaluate. 
# Part-1 Decision Tree Regressor Model

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load and preprocess data
file = pd.read_excel('ML/emzirme082023_v4.xlsx')
file.drop("gebelik_tipi_gruplu", axis=1, inplace=True)

# Rename the columns: using camelHump notation
file.rename(columns= {"bebekadi": "bebekAdi", "tanı_gruplu" : "taniGruplu", "tanısı" : "tanisi",
            "dogum_agırlıgı_gruplu": "dogumAgirligiGruplux", "dogumagirligi(gram)" : "dogumAgirligiGram", 
            "taburculuktakilogram": "taburculuktakiKilogram", "takipilkgün_kilo_gram": "takipIlkGunKiloGram",
            "dogumtarihi": "dogumTarihi", "bebek_dostu_20temmuz2018": "bebekDostu20Temmuz2018",
            "emzirmeyebasladigitarih": "emzirmeyeBasladigiTarih", "takibegirdigigun": "takibeGirdigiGun",
            "takibegirdigitarih": "takibeGirdigiTarih", "taburculuktarihi": "taburculukTarihi",
            "gebeliktipi": "gebelikTipi", "gebelik_34": "gebelik34", "gebelik_haftası_gruplu": "gebelikHaftasiGruplu",
            "gebelikhaftası": "gebelikHaftasi", "gebelikhaftagunu": "gebelikHaftaGunu", "dogumyeri": "dogumYeri",
            "anneadi": "anneAdi", "anne_meslek_grup": "anneMeslekGrup", "annemeslegi": "anneMeslegi", "anne_yaşı_grup": "anneYasiGrup",
            "anneyasi": "anneYasi", "anne_egitim_grup": "anneEgitimGrup", "anneegitim": "anneEgitim", "dogumsekli": "dogumSekli", 
            "yasayancocuksayisi": "yasayanCocukSayisi", "emzirdigicocuksayisi": "emzirdigiCocukSayisi", "bironcekibebegikacayemzirdi" : "birOncekiBebegiKacAyEmzirdi",
            "anne_hastalık_grup": "anneHastalıkGrup", "Anne_Hastalık": "anneHastalik", "Kullandıgıilaclar": "kullandigiIlaclar", "Kullandıgıpompamarkasi": "kullandigiPompaMarkasi",
            "Kullandıgıpompamarkasi": "kullandigiPompaMarkasi", "Kolostrumvarligi": "kolostrumVarligi", "takiptekacgun": "takipteKacGun", "takiptekacgun": "takipteKacGun",
            "beslenmetotali1.guncc_alması_gereken": "beslenmeTotali1.GunCCAlmasiGereken", "70cc-60cc bağlantılı kg carpımı sonuc": "70CC-60CCBaglantiliKgCarpimiSonuc",
            "1.gun totali": "1.GunTotali", "1.gun totali": "1.GunTotali", "70cc-60cc bağlantılı kg carpımı sonuçun simgesel gösterimi": "70CC-60CCBaglantiliKgCarpimiSonucunSimgeselGosterimi",
            "aldığımamamiktari1.gün": "aldigiMamaMiktari1.Gun", "ilk_gün_anne_sütü_1111": "ilkGunAnneSutu1111", "ilk_gün_anne_sütü_1111": "ilkGunAnneSutu1111", "ilkgün_bebeğinannesütüalımı": "ilkGunBebeginAnneSutuAlimi",
            "aldığıannesütü_ilkgün": "aldigiAnneSutuIlkGun", "beslenmetotali2.gün": "beslenmeTotali2.Gun", "beslenme totali 2.gün cckg sonucu": "beslenmeTotali2.GunCCKGSonucu", "beslenmemamamiktarı2.guncc": "beslenmeMamaMiktari2.GunCC",
            "beslenme2.gunannesutucc": "beslenme2.GunAnneSutuCC", "beslenmetotali3.gun": "beslenmeTotali3.Gun", "aldıgımamamiktari3.gun": "aldigiMamaMiktari3.Gun", "aldıgıannesütü3.gun": "aldigiAnneSutu3.Gun",
            "beslenmeninilkgunuverilisyolu": "beslenmeninIlkGunuVerilisYolu", "beslenmetotali3.gun": "beslenmeTotali3.Gun", "aldıgımamamiktari3.gun": "aldigiMamaMiktari3.Gun", "aldıgıannesütü3.gun": "aldigiAnneSutu3.Gun",
            "ilk_gün_emzirme_111": "ilkGunEmzirme111", "verilisyolu2.gun": "verilisYolu2.Gun", "verilisyolu3gun": "verilisYolu3Gun", "beslenmetotalitaburculuk": "beslenmeTotaliTaburculuk",
            "taburculuktamamamiktari": "taburculuktaMamaMiktari", "aldığıannesütü_taburculuk": "aldigiAnneSutuTaburculuk", "taburculukta_annesutu_111": "taburculuktaAnneSutu111", "taburculuk_beslenmeturu": "taburculukBeslenmeTuru",
            "taburculuktanasılbeslenmeyolu": "taburculuktaNasilBeslenmeYolu", "kacgunogkullandi": "kacGunOGKullandi", "taburculuktaogvarmiyokmu": "taburculuktaOGVarMiYokMu", "baslangictasutdestegi": "baslangictaSutDestegi",
            "sutdestegivarsakacolcek": "sutDestegiVarsaKacOlcek", "varsataburculuktakaçölçek": "varsaTaburculuktaKacOlcek", "kacıncıgundesutdestegibaslandı": "kacinciGundeSutDestegiBaslandi", "taburculuktadestekcesidi": "taburculuktaDestekCesidi",
            "annesutuemzirmeeğitimidurumu": "anneSutuEmzirmeEgitimiDurumu", "galaktokogkullanımı": "galaktokogKullanimi", "memesorunuyaşamadurumu": "memeSorunuYasamaDurumu", "memesorunuvarsa_tedavidekullanılanlar": "memeSorunuVarsaTedavideKullanilanlar",
            "taburculukrtasutdestegivarmı": "taburculuktaSutDestegiVarMi", "Ataburculuktaannesutu": "taburculuktaAnneSutu", "emzirme_Taburculuk": "emzirmeTaburculuk", "covid19sonrasi": "covid19Sonrasi", "Postnatalgunemzirme": "postNatalGunEmzirme",
            "pntakibegirdigitarih": "pnTakibeGirdigiTarih", "pntaburculuktarihi": "pnTaburculukTarihi", "bebekdostuoncesonra": "bebekDostuOnceSonra", "ikisiarası": "ikisiArasi"
            }, inplace=True)

# Converting numbers stored as text to numeric and handle null values
file["anneHastalik"] = pd.to_numeric(file["anneHastalik"], errors='coerce')
file["kullandigiIlaclar"] = pd.to_numeric(file["kullandigiIlaclar"], errors='coerce')

file["sutDestegiVarsaKacOlcek"].fillna(0, inplace=True)
file["varsaTaburculuktaKacOlcek"].fillna(0, inplace=True)
file["taburculuktaDestekCesidi"].fillna(0, inplace=True)

# Now, save the modified DataFrame back to an Excel file
file.to_excel('ML/emzirme082023_v4.xlsx', index=False)