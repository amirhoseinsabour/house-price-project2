# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 21:01:25 2025

@author: Elecomp
"""

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([[50, 2, 10, 1],
              [80, 3, 5, 3],
              [120, 4, 15, 2],
              [60, 2, 8, 1],
              [90, 3, 12, 2]])

y = np.array([500, 900, 1500, 600, 1000])


X = X / X.max(axis=0)

y = y / y.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# تعریف مدل
model = Sequential([
    Dense(5, input_dim=X_train.shape[1], activation='relu'),  
    Dense(1)  
])

model.compile(optimizer='adam', loss='mse')  


history = model.fit(X_train, y_train, epochs=200, verbose=1)



y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("پیش‌بینی روی Train:", y_train_pred.flatten())
print("پیش‌بینی روی Test:", y_test_pred.flatten())


#امیرحسین صبور دربندی  
# هوش مصنوعی شنبه ساعت ده شماره داشنجویی 40111141052203