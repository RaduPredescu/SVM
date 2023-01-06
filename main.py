from sklearn import svm
import numpy as np
import array
import csv
import pandas

data = pandas.read_csv("LasVegasTripAdvisorReviews-Dataset.csv", delimiter=';')

# facem imparteala intre date si etichete
classes = data["Score"]
attributes = data.loc[:, data.columns != 'Scores']


# calculam dimensiunile
sz = len(classes)

# print(sz)


trainSize = (int)(0.75*sz)
testSize = (int)(0.25*sz)


# facem imparteala intre date de test si date de antrenare(cred)


attributesTrain = attributes[:trainSize]
attributesTest = attributes[trainSize:]


classesTrain = classes[:trainSize]
classesTest = classes[trainSize:]


# construim matricea de costuri(nu stiu de ce merge asa, dar probabil merge)

costMatrix = list(range(-7, 6))
costMatrix = 2.**np.array(costMatrix)

# print(attributesTrain) aici nu stiu de ce fac asta, dar nu stiu ce altceva sa mai scriu ca am prea multe erori

for i in range(len(costMatrix)):
    ans = svm.SVC(kernel='linear', C=costMatrix[i]).fit(attributesTrain, classesTrain)
    predictionsTest = ans.predict(attributesTest)
    correctPredictions = 0
    for j in range(0, len(predictionsTest)):
        if (predictionsTest[j] == classesTest[j]):
            correctPredictions = correctPredictions+1
    print('Acuratete pentru cost 2^' + str(i-7) + ': ' + str(correctPredictions/testSize) +
          ', ' + str(correctPredictions) + ' predictii corecte din ' + str(testSize))
