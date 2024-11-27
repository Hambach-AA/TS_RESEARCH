import pickle

ts = {}

"""
name - Имя ряда
dimension - Измерение ряда
subseqLen - длина подпоследовательности ряда
fragmentLen - длина фрагмента ряда
fragmentNum - номер фрагмента ряда
"""

ts['ThermalSensor_1'] = {
    'name' : 'ThermalSensor',
    'dimension' : '1',
    'subseqLen' : 256,
    'fragmentLen' : 1000,
    'fragmentNum' : 1}
    
ts['ThermalSensor_2'] = {
    'name' : 'ThermalSensor',
    'dimension' : '2',
    'subseqLen' : 256,
    'fragmentLen' : 1000,
    'fragmentNum' : 1}
    
with open('configTS', 'wb') as f:
    pickle.dump(ts, f)
