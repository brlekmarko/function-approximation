import math
import sys

def readInput(fileName):
    data = []
    ## prva linija data liste je zaglavlje
    with open(fileName, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data

class Leaf:
    def __init__(self, vrijednost):
        self.vrijednost = vrijednost
    
    def isLeaf(self):
        return True


class Node:
    def __init__(self, znacajka, subtrees):
        self.znacajka = znacajka
        self.subtrees = subtrees
    
    def isLeaf(self):
        return False
    

class ID3:
    def __init__(self, dubina = -1):
        self.dubina = dubina
        self.tree = None
        
    def fit(self, train_dataset):
        global headers
        self.train_dataset = train_dataset
        toReturn = self.id3(train_dataset, train_dataset, headers[:-1], headers[-1], self.dubina)
        self.tree = toReturn

    def predict(self, test_dataset):
        global headers

        ### PRINTAMO GRANE ###
        print("[BRANCHES]:")
        self.printBranches(self.tree, "")

        ### RADIMO PREDVIĐANJE ###
        rezultati = []
        ispis = "[PREDICTIONS]:"
        for test in test_dataset:
            rezultat = self.predictTest(self.tree, test)
            rezultati.append(rezultat)
            ispis += " " + rezultat
        print(ispis)

        ### RACUNAMO TOCNOST ###
        correctAnswers = []
        for test in test_dataset:
            correctAnswers.append(test[-1])

        matchingAnswers = 0
        for i in range(len(correctAnswers)):
            if correctAnswers[i] == rezultati[i]:
                matchingAnswers += 1

        accuracy = matchingAnswers / len(correctAnswers)
        print("[ACCURACY]: " + '{0:.5f}'.format(accuracy))


        ### RACUNAMO MATRICU ZABUNE ###
        print("[CONFUSION_MATRIX]:")

        ## prvo moramo naci sve ciljeve koji se pojavljuju u test_datasetu
        moguciCiljevi = set()
        for test in test_dataset:
            moguciCiljevi.add(test[-1])
        moguciCiljevi = list(moguciCiljevi)
        moguciCiljevi.sort()

        ## inicijaliziramo matricu zabune
        ## velicine je len(moguciCiljevi) x len(moguciCiljevi)
        ## inicijalno su sve vrijednosti 0
        matricaZabune = []
        for i in range(len(moguciCiljevi)):
            matricaZabune.append([0] * len(moguciCiljevi))

        
        ## popunjavamo matricu
        for i in range(len(correctAnswers)):
            redak = moguciCiljevi.index(correctAnswers[i])
            stupac = moguciCiljevi.index(rezultati[i])
            matricaZabune[redak][stupac] += 1
        
        ## ispisujemo matricu
        for redak in range(len(moguciCiljevi)):
            ispis = ""
            for stupac in range(len(moguciCiljevi)):
                ispis += str(matricaZabune[redak][stupac]) + " "
            print(ispis)


    ## algoritam sa prezentacije
    ## tests = testni skup
    ## tests_parent = roditeljski skup testova
    ## znacajke = skup znacajki
    ## ciljna_znacajka = ciljna znacajka
    ## dubina = maksimalna dubina stabla, ako je -1, onda je dubina neogranicena
    ##          sa svakim rekurzivnim pozivom se smanjuje za 1, pa ako je 0, vracamo list
    def id3(self, tests, tests_parent, znacajke, ciljna_znacajka, dubina = -1):
        if len(tests) == 0:
            vrijednost = argmaxSize(tests_parent, ciljna_znacajka)
            return Leaf(vrijednost)
        vrijednost = argmaxSize(tests, ciljna_znacajka)
        if dubina == 0:
            return Leaf(vrijednost)
        if len(znacajke) == 0 or tests == filterTests(tests, ciljna_znacajka, vrijednost):
            return Leaf(vrijednost)
        najbolja_znacajka = argmaxInformationGain(tests, znacajke)
        subtrees = set()
        for value in getValues(najbolja_znacajka):
            Dxv = filterTests(tests, najbolja_znacajka, value)
            nove_znacajke = znacajke.copy()
            nove_znacajke.remove(najbolja_znacajka)
            podstablo = self.id3(Dxv, tests, nove_znacajke, ciljna_znacajka, dubina - 1)
            subtrees.add((value, podstablo))
        return Node(najbolja_znacajka, subtrees)

    def printBranches(self, node, path, level=1):
        if node.isLeaf():
            ## ako je list, ispisujemo vrijednost
            pathString = path + node.vrijednost
            print(pathString)
        else:
            ## inace rekurzivno pozivamo za svako podstablo
            for value, subtree in node.subtrees:
                pathString = path + str(level) + ":" + node.znacajka + "=" + value + " "
                self.printBranches(subtree, pathString, level + 1)
    
    def predictTest(self, node, test):
        global headers
        if node.isLeaf():
            ## dosli smo do dna stabla, vracamo vrijednost
            return node.vrijednost
        ## inace, u cvoru smo koji ima znacajku i podstabla
        znacajka = node.znacajka
        index = headers.index(znacajka) ## index znacajke u testu
        trazena_vrijednost = test[index]
        for vrijednost, subtree in node.subtrees:
            if vrijednost == trazena_vrijednost:
                ## idemo u podstablo koje odgovara vrijednosti
                return self.predictTest(subtree, test)
        ## susreo se s vrijednosti koju do sad nije susreo, vracamo najcescu vrijednost
        currentTests = self.train_dataset
        for i in range(index-1):
            currentTests = filterTests(currentTests, headers[i], test[i])
        return argmaxSize(currentTests, headers[-1])
    

############# POMOCNE METODE #############
##########################################   
    
## vraca vrijednost atributa y koja se najcesce pojavljuje u testovima
def argmaxSize(testovi, znacajka):
    max, maxVrijednost = 0, 0
    for vrijednost in getValues(znacajka):
        filtrirani_testovi = filterTests(testovi, znacajka, vrijednost)
        size = len(filtrirani_testovi)
        if size >= max:
            ## ako je velicina veca ili jednaka, uzimamo vrijednost koja je abecedno prva
            if size == max and maxVrijednost != 0 and vrijednost > maxVrijednost:
                continue
            max, maxVrijednost = size, vrijednost
    return maxVrijednost

## vraca rjecnik gdje su kljucevi vrijednosti atributa, a vrijednosti skup svih vrijednosti koje se pojavljuju u data
def calculateValues(tests):
    global headers
    vrijednosti = {}
    for i in range(len(headers)):
        vrijednosti[headers[i]] = set()
        for test in tests:
            vrijednosti[headers[i]].add(test[i]) ## dodajemo vrijednost u skup, u mapu pod kljucem znacajke
    return vrijednosti

## vraca sve vrijednosti atributa x koji se pojavljuju u data
## vrijednosti su prethodno izracunate u calculateValues
def getValues(x):
    global values
    return values[x]

## vraca samo one retke iz data koji imaju zadanu vrijednost za zadanu znacajku
def filterTests(tests, znacajka, vrijednost):
    filteredTests = []
    for test in tests:
        index = headers.index(znacajka)
        if test[index] == vrijednost:
            filteredTests.append(test)
    return filteredTests

## vraca znacajku koja ima najveci dobit inforamcije
def argmaxInformationGain(testovi, znacajke):
    max, maxZnacajka = 0, 0
    ##ispis = ""
    for znacajka in znacajke:
        currentGain = calculateInformationGain(testovi, znacajka)
        ##ispis += "IG(" + znacajka + ") = " + '{0:.4f}'.format(currentGain) + " "
        if currentGain >= max:
            ## ako su jednako dobre, uzimamo onu koja je abecedno prva
            if currentGain == max and maxZnacajka != 0 and znacajka > maxZnacajka:
                continue
            max, maxZnacajka = currentGain, znacajka
    ##print(ispis)
    return maxZnacajka

## racuna dobit informacije za znacajku x
def calculateInformationGain(testovi, znacajka):
    global values
    informationGain = calculateEntropy(testovi)
    for vrijednost in getValues(znacajka):
        ## idemo po svim vrijednostima zadane znacajke
        filtrirani = filterTests(testovi, znacajka, vrijednost)
        informationGain -= (len(filtrirani) / len(testovi)) * calculateEntropy(filtrirani)
    return informationGain

## racuna entropiju skupa testova
def calculateEntropy(testovi):
    global headers
    rezultatZnacajka = headers[-1]
    moguciRezultati = getValues(headers[-1])
    entropija = 0
    for rezultat in moguciRezultati:
        filtrirani = filterTests(testovi, rezultatZnacajka, rezultat)
        if len(filtrirani) > 0:
            omjer = len(filtrirani) / len(testovi)
            entropija -= omjer * math.log(omjer, 2)
    return entropija



##### MAIN #####
################
if __name__ == '__main__':
    numOfArgs = sys.argv[0]
    args = sys.argv[1:]
    dataFile = args[0]
    testDataFile = args[1]
    depth = -1
    if len(args) > 2:
        depth = int(args[2])

    data = readInput(dataFile)
    headers = data[0]
    data = data[1:]
    values = calculateValues(data)

    id3 = ID3(depth)
    id3.fit(data)

    testData = readInput(testDataFile)
    testData = testData[1:]

    id3.predict(testData)