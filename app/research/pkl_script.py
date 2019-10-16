import pickle

with open(r'./resources/pkl/cells2_2016-03-01_21.42.11.pkl','rb') as file:
    contours = pickle.load(file)

print(contours)