import numpy as np
import math
import random
import sys

def loadInputFile(fileName):
    data = []
    ## prva linija u data su headeri
    with open(fileName, 'r') as f:
        for line in f:
            splitano = line.split(',')
            if (len(data)>0):
                for i in range(len(splitano)):
                    splitano[i] = float(splitano[i])
            data.append(splitano)
    return data

def obradi(data, dimensions, functions, primjerak_populacije):
    # propagiraj sve testove iz data do kraja
    # vrati pogresku
    rezultati = propagirajUnaprijed(dimensions, primjerak_populacije.matricaTezina, primjerak_populacije.matricaOsjetljivosti, data, functions)
    pogreska = izracunajPogresku(data, rezultati)
    return pogreska

class Primjerak_Populacije:
    ## matricaTezina oblikovana na nacin da je tezina iz cvora 1 iz prvog sloja do cvora 3 iz drugog sloja
    ## = matricaTezina[0][0][2]

    ## matricaOsjetljivosti oblikovana nacin da je osjetljivost cvora 2 u drugom sloju = matricaOsjetljivosti[1][1]
    def __init__(self, dimensions=None):
        if(dimensions == None):
            self.matricaTezina = None
            self.matricaOsjetljivosti = None
        else:
            self.matricaTezina = kreirajRandomMatricuTezine(dimensions)
            self.matricaOsjetljivosti = kreirajRandomMatricuOsjetljivosti(dimensions)



def propagirajUnaprijed(dimensions, matricaTezina, matricaOsjetljivosti, data, funkcije):
    ## propagira ulaz testova kroz neuronsku mrezu do izlaza
    ## vraca vrijednosti izlaza za svaki test
    ## koristen algoritam s predavanja s mnozenjem matrica
    koncaniRezultati = []
    for u in range(1, len(data)):
        rezultati = []
        for i in range(len(dimensions)-1):
            tezine = np.array(matricaTezina[i]).transpose()
            if(i == 0):
                podaci = np.array(data[u][:-1]).transpose()
            else:
                podaci = rezultati[i-1]
            osjetljivosti = np.array(matricaOsjetljivosti[i+1]).transpose()
            rezultat = np.add(np.matmul(tezine, podaci), osjetljivosti)
            if(i<len(funkcije)):
                for j in range(len(rezultat)):
                    ## primjenimo funkciju
                    ## s = sigmoida
                    if(funkcije[i] == "s"):
                        rezultat[j] = 1/(1+math.pow(math.e, -rezultat[j]))
            rezultati.append(rezultat)
        koncaniRezultati.append(rezultati[-1][0])
    return koncaniRezultati

def izracunajPogresku(data, rezultati):
    ## usporeduje nase dobivene rezultate sa zadnjim stupcem iz data (tocni rezultati)
    ## vraca srednju sumu razlike kvadrata
    tocniRezultati = []
    for i in range(1, len(data)):
        tocniRezultati.append(data[i][-1])
    zbrojRazlikeKvadrata = 0
    for i in range(len(rezultati)):
        zbrojRazlikeKvadrata += math.pow(tocniRezultati[i]-rezultati[i],2)
    toReturn = (1/len(rezultati))*zbrojRazlikeKvadrata
    return toReturn

def kreirajRandomMatricuTezine(dimensions):
    ## matrica oblikovana na nacin da je tezina iz cvora 1 iz prvog sloja do cvora 3 iz drugog sloja
    ## = matricaTezina[0][0][2]
    ulaznaDimenzija = dimensions[0]
    srednjeDimenzije = dimensions[1:-1]
    izlaznaDimenzija = dimensions[-1]
    matricaTezina = []
    tezinaSloj = []
    for i in range (ulaznaDimenzija):
        ## broj tezina za svaki neuron s lijeve strane je broj neurona s desne strane
        tezinaSloj.append(np.random.normal(0, 0.01, srednjeDimenzije[0]))
    matricaTezina.append(tezinaSloj)
    for i in range (len(srednjeDimenzije)):
        tezinaSloj = []
        if(i == len(srednjeDimenzije)-1):
            for j in range(srednjeDimenzije[i]):
                ## broj tezina za svaki neuron s lijeve strane je broj neurona na izlazu (1)
                tezinaSloj.append(np.random.normal(0, 0.01, izlaznaDimenzija))
        else:
            for j in range(srednjeDimenzije[i]):
                ## broj tezina za svaki neuron s lijeve strane je broj neurona s desne strane
                tezinaSloj.append(np.random.normal(0, 0.01, srednjeDimenzije[i+1]))
        matricaTezina.append(tezinaSloj)
    return matricaTezina

def kreirajRandomMatricuOsjetljivosti(dimensions):
    ## matricaOsjetljivosti oblikovana nacin da je osjetljivost cvora 2 u drugom sloju = matricaOsjetljivosti[1][1]
    ulaznaDimenzija = dimensions[0]
    srednjeDimenzije = dimensions[1:-1]
    izlaznaDimenzija = dimensions[-1]
    matricaOsjetljivosti = []

    matricaOsjetljivosti.append([0] * ulaznaDimenzija) ## osjetljivost ulaznih je 0
    for i in range(len(srednjeDimenzije)):
        ## broj "osjetljivosti" je velicina dimenzije
        matricaOsjetljivosti.append(np.random.normal(0, 0.01, srednjeDimenzije[i]))
    matricaOsjetljivosti.append(np.random.normal(0, 0.01, izlaznaDimenzija))
    
    return matricaOsjetljivosti

def genetskiAlgoritam(data, dimensions, functions, velPop, elitizam, vjer_mutacija, skala_mutacije, broj_iteracija):
    ## pracen algoritam s predavanja
    ## prvo generira nasumicnu populaciju
    ## zatim prema ruletu selektira parove iz generacije te ih kriza
    ## krizanje se radi uzimanjem aritmeticke sredine
    ## nakon krizanja mutira, svaki kromosom ima vjer_mutacija vrijednosti da mutira
    ## mutira se tako da se kromosomu pribroji vrijednost iz normalne razdiobe sa srednjom devijacijom skala_mutacije
    ## radi se broj_iteracija iteracija
    populacija = []
    rezultati = []
    sumaRezultata = 0
    ## generiramo nasumicnu populaciju
    for i in range(velPop):
        noviPrimjerak = Primjerak_Populacije(dimensions) ## ovo generira nasumican primjerak
        populacija.append(noviPrimjerak)
        rezultat = 1/obradi(data, dimensions, functions, noviPrimjerak)
        sumaRezultata += rezultat
        rezultati.append(rezultat)

    for i in range(broj_iteracija):
        if(i>0 and i%2000==0):
            print("[Train error @" + str(i) + "]: " + "{:.6f}".format(1/max(rezultati))) ##svakih 2000 iteracija ispisuje
        nova_populacija = []
        
        ## uzima elitizam najboljih jedinki i odmah ih doda u novu populaciju
        if(elitizam == 1):
            nova_populacija.append(populacija[rezultati.index(max(rezultati))])
        else:
            sortirani = sorted(rezultati, reverse=True)
            for j in range(elitizam):
                nova_populacija.append(populacija[rezultati.index(sortirani[j])])

        while len(nova_populacija) < velPop:
            zaKrizanje1, zaKrizanje2 = nadjiKandidateZaKrizanje(populacija, rezultati, sumaRezultata) ## metodom ruleta
            krizani1, krizani2 = krizaj(zaKrizanje1, zaKrizanje2), krizaj(zaKrizanje1, zaKrizanje2) ## aritmeticka sredina
            mutiraj(krizani1, vjer_mutacija, skala_mutacije) ## random
            mutiraj(krizani2, vjer_mutacija, skala_mutacije)
            nova_populacija.append(krizani1)
            nova_populacija.append(krizani2)

        populacija = nova_populacija
        rezultati = []
        sumaRezultata = 0
        ## evaluiramo novu populaciju
        for i in range(velPop):
            rezultat = 1/obradi(data, dimensions, functions, populacija[i])
            sumaRezultata += rezultat
            rezultati.append(rezultat)

    maxRez = max(rezultati)
    print("[Train error @" + str(broj_iteracija) + "]: " + "{:.6f}".format(1/maxRez))
    pobjednik = populacija[rezultati.index(maxRez)]
    return pobjednik

def mutiraj(primjerak_populacije, vjer_mutacija, skala_mutacije):
    ## dodji do kromosoma, "baci kockicu", dodaj mu random vrijednosti
    for i in range(len(primjerak_populacije.matricaTezina)):
        for j in range(len(primjerak_populacije.matricaTezina[i])):
            ## probamo mutirati
                randomBroj = random.random()
                if(randomBroj < vjer_mutacija):
                    ## mutiramo
                    primjerak_populacije.matricaTezina[i][j] += np.random.normal(0, skala_mutacije, len(primjerak_populacije.matricaTezina[i][j]))
    for i in range(len(primjerak_populacije.matricaOsjetljivosti)):
        ## probamo mutirati
            randomBroj = random.random()
            if(randomBroj < vjer_mutacija):
                ## mutiramo
                primjerak_populacije.matricaOsjetljivosti[i] += np.random.normal(0, skala_mutacije, len(primjerak_populacije.matricaOsjetljivosti[i]))
    return


def nadjiKandidateZaKrizanje(populacija, rezultati, sumaRezultata):
    ## sto je primjerak populacije imao bolji rezultat, to je veca vjerojatnost da ga izabere
    randomBroj = random.random()
    toReturn1, toReturn2 = None, None
    ## iteriramo, smanjujemo randomBroj za velicinu rezultata, kad se smanji ispod nule izaberemo taj
    for i in range(len(rezultati)):
        vjerojatnost_krizanja = (rezultati[i]/sumaRezultata)
        randomBroj -= vjerojatnost_krizanja
        if(randomBroj<=0):
            toReturn1 = populacija[i]
            break

    ## isto kao i gore jos jedamput
    randomBroj = random.random()
    for i in range(len(rezultati)):
        vjerojatnost_krizanja = (rezultati[i]/sumaRezultata)
        randomBroj -= vjerojatnost_krizanja
        if(randomBroj<=0):
            toReturn2 = populacija[i]
            break
    return toReturn1, toReturn2

def krizaj(primjerakPopulacije1, primjerakPopulacije2):
    ## zbrojimo matrice i podijelimo ih s 2
    novaMatricaTezina = []
    for i in range(len(primjerakPopulacije1.matricaTezina)):
        ## ovaj for je potreban jer inace nisu svi array-evi iste velicine pa se numpy zali
        novaMatricaTezina.append(np.add(primjerakPopulacije1.matricaTezina[i], primjerakPopulacije2.matricaTezina[i])/2)

    novaMatricaOsjetljivosti = []
    for i in range(len(primjerakPopulacije1.matricaOsjetljivosti)):
        novaMatricaOsjetljivosti.append(np.add(primjerakPopulacije1.matricaOsjetljivosti[i], primjerakPopulacije2.matricaOsjetljivosti[i])/2)

    noviPrimjerak = Primjerak_Populacije(None)
    noviPrimjerak.matricaTezina = novaMatricaTezina
    noviPrimjerak.matricaOsjetljivosti = novaMatricaOsjetljivosti
    return noviPrimjerak


args = sys.argv[1:]

trainFileName = ""
testFileName = ""
arhitektura = ""
popsize = 0
elitism = 0
vjer_mutacije = 0
skala_mutacije = 0
broj_iteracija = 0
for i in range(len(args)):
    if(args[i] == "--train"):
        trainFileName = args[i+1]
        i+=1
        continue
    if(args[i] == "--test"):
        testFileName = args[i+1]
        i+=1
        continue
    if(args[i] == "--nn"):
        arhitektura = args[i+1]
        i+=1
        continue
    if(args[i] == "--popsize"):
        popsize = int(args[i+1])
        i+=1
        continue
    if(args[i] == "--elitism"):
        elitism = int(args[i+1])
        i+=1
        continue
    if(args[i] == "--p"):
        vjer_mutacije = float(args[i+1])
        i+=1
        continue
    if(args[i] == "--K"):
        skala_mutacije = float(args[i+1])
        i+=1
        continue
    if(args[i] == "--iter"):
        broj_iteracija = int(args[i+1])
        i+=1
        continue

data = loadInputFile(trainFileName)

headers = data[0]
data = data[1:]
ulaznaDimenzija = len(headers)-1
izlaznaDimenzija = 1
dimensions = [ulaznaDimenzija]
functions = []
broj = 0
for i in range(0,len(arhitektura)):
    if(arhitektura[i]>='0' and arhitektura[i]<='9'):
        broj*=10
        broj+=int(arhitektura[i])
    else:
        dimensions.append(broj)
        functions.append(arhitektura[i])
        broj=0
dimensions.append(izlaznaDimenzija)

pobjednik = genetskiAlgoritam(data, dimensions, functions, popsize, elitism, vjer_mutacije, skala_mutacije, broj_iteracija)

testData = loadInputFile(testFileName)
data = data[1:]

testGreska = obradi(testData, dimensions, functions, pobjednik)
print("[Test error]: {:.6f}".format(testGreska))
