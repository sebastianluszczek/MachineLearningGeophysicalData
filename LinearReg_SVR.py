import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import WykresClass
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle


# importujemy dane geofizyki otworowej, z pliku xlsx, każdy Arkusz odpowiada jednemu otworowi
with pd.ExcelFile('dane otworowe.xlsx') as xls:
    L1_df = pd.read_excel(xls, 'L-1')
    K1_df = pd.read_excel(xls, 'K-1')
    O2_df = pd.read_excel(xls, 'O-2')

# każdy otwór zapisany do odręnej df, głębokość pomiarów indexem
L1_df.set_index('DEPT', inplace=True)
K1_df.set_index('DEPT', inplace=True)
O2_df.set_index('DEPT', inplace=True)


# dane pomiarowe składają się z blisko 30 kolumn z danymi, do badań wystarczy tylko podstawowe 12 (+DEPT)
L1_df = L1_df[['LLS', 'LLD', 'DT', 'RHOB', 'Pe', 'NPHI', 'GR', 'GKUT', 'GRKT', 'POTA',
       'THOR', 'URAN']]
K1_df = K1_df[['LLS', 'LLD', 'DT', 'RHOB', 'Pe', 'NPHI', 'GR', 'GKUT', 'GRKT', 'POTA',
       'THOR', 'URAN']]
O2_df = O2_df[['LLS', 'LLD', 'DT', 'RHOB', 'Pe', 'NPHI', 'GR', 'GKUT', 'GRKT', 'POTA',
       'THOR', 'URAN']]

# usuwamy wiersze z brakującymi danymi
K1_df.dropna(inplace=True)
O2_df.dropna(inplace=True)
# w df L1_df cała kolumna 'DT' ma brak danych, stąd trzeba było wykluczyć ja z polecenia pd.df.dropna()
L1_df.dropna(subset=['LLS', 'LLD', 'RHOB', 'Pe', 'NPHI', 'GR', 'GKUT', 'GRKT', 'POTA',
       'THOR', 'URAN'], inplace=True)

# rysuje wykresy porównujące 3 otwory, na podstawie podstawowych krzywych: DT, RHOB, NPHI
# (L1 nie zawiera DT, a cały projekt docelowo ma modelować jego przebieg w L1)
# do rysowania wykresów urzywam napisanej przez siebie klasy WykresClass i na razie jej jedynem metody WykresyClass.rysuj()
wykr = WykresClass.WykresyClass(list=[[L1_df, 'DT', 'RHOB', 'NPHI'],
                                      [K1_df, 'DT', 'RHOB', 'NPHI'],
                                      [O2_df, 'DT', 'RHOB', 'NPHI']], save='wykresy_przed')
wykr.rysuj()

# tworzę zbiory krzywych wejściowych modelu i krzywej wynikowej (projekt ma na celu modelowanie DT)
X_K1 = np.array(K1_df.drop(['DT'], 1))
y_K1 = np.array(K1_df['DT'])

X_O2 = np.array(O2_df.drop(['DT'], 1))
y_O2 = np.array(O2_df['DT'])

# na razie pomijam skalowanie ze względu na późniejsze problemy z przewidywaniem dla nowych danych,
# dokłądniej się nim zajmę w przyszłosci
# X = preprocessing.scale(X)
# y = preprocessing.scale(y)

# crosswalidacja zbiorów odpowiednio dla otworów K-1 i O-2
X_K1_train, X_K1_test, y_K1_train, y_K1_test = cross_validation.train_test_split(X_K1, y_K1, test_size=0.2)
X_O2_train, X_O2_test, y_O2_train, y_O2_test = cross_validation.train_test_split(X_O2, y_O2, test_size=0.2)

# definiujemy 'klasyfikatory', obecnie mogę porównać sposoby: LinearRegresion i SVR
clf1 = LinearRegression(n_jobs=-1)
clf2 = LinearRegression(n_jobs=-1)

# tworzenie modelu dla otworu K-1
clf1.fit(X_K1_train, y_K1_train)
confidence1 = clf1.score(X_K1_test, y_K1_test)

# tworzenie modelu dla otworu K-1
clf2.fit(X_O2_train, y_O2_train)
confidence2 = clf2.score(X_O2_test, y_O2_test)

# porównanie wyników dla różnych modeli
# print('LinearRegression classificator have confidence: {} on BH K-1 and confidence: {} on BH O-2'.format(confidence1, confidence2))
#for k in ['linear','poly','rbf','sigmoid']:
#    clf1 = svm.SVR(kernel=k)
#    clf1.fit(X_K1_train, y_K1_train)
#    confidence2 = clf.score(X_K1_test, y_K1_test)

#    clf2 = svm.SVR(kernel=k)
#    clf2.fit(X_O2_train, y_O2_train)
#    confidence2 = clf.score(X_O2_test, y_O2_test)
#    print('SVR (kernel=[{}]) classificator have confidence: {} on BH K-1 and confidence: {} on BH O-2''.format(k, confidence1, confidence2))

# zapisujemy clasyfikatory jako pickle, żeby nie musieć przeprowadzać przy koażdej prubie tego samego procesu uczenia
with open('classificator_Lin_Reg_K-1.pickle', 'wb') as f:
    pickle.dump(clf1, f)
with open('classificator_Lin_Reg_O-2.pickle', 'wb') as f:
    pickle.dump(clf2, f)

# odczytujemy pickle
pickle_in = open('classificator_Lin_Reg_K-1.pickle', 'rb')
clf1 = pickle.load(pickle_in)
pickle_in = open('classificator_Lin_Reg_O-2.pickle', 'rb')
clf2 = pickle.load(pickle_in)

# tworzymy df'y które posłużą do próby zamodelowania wart. kolumny 'DT' w poszczególnych Otworach
X_O2_predict = np.array(O2_df.drop(['DT'], 1))
X_K1_predict = np.array(K1_df.drop(['DT'], 1))

# na razie bez skalowania
#X_O2_predict = preprocessing.scale((X_O2_predict))
#X_K1_predict = preprocessing.scale((X_K1-predict))

# przewidujemy wartość modelowanego 'DT' w otworze K-1 na podstawie modelu dla O-2 i na odwrót
DT_K1_pred = clf2.predict(X_K1_predict)
DT_O2_pred = clf1.predict(X_O2_predict)

# dodajemy kolumny z wymodelowanym DT-predicted do DataFramów z naszymi danymi otworowymi
O2_df['DT_predicted_LR'] = DT_O2_pred
K1_df['DT_predicted_LR'] = DT_K1_pred

# w wydruk w ramach wstępnego sprawdzenia
print('K-1', '\n', K1_df[['DT', 'DT_predicted_LR']].describe())
print('O-2', '\n', O2_df[['DT', 'DT_predicted_LR']].describe())

# ponownie za pomocą utworzonej klasy rysujemy wykresy, tym razem w celu zestawienia DT rzeczywistego i zamodelowanego
# dla obu otworów
wykr2 = WykresClass.WykresyClass(list=[[K1_df, 'DT', 'DT_predicted_LR'], [O2_df, 'DT', 'DT_predicted_LR']],
                                 save='wykresy_LR', x_scale_static=True)
wykr2.rysuj()


