import contestLib as cl
from words import program

#dupa = toMaSieWysypac_maszTegoNieWYykonywac

prices, rooms, meters, floors, pricesPerMeter, locations, descriptions = cl.loadTrainData()
text = ' '.join(descriptions)
diction = program.getNMostFrequentWords(text, 1000000)
cl.saveObj(diction, "traindict")

rooms, meters, floors, locations, descriptions = cl.loadDevData()
text = ' '.join(descriptions)
diction = program.getNMostFrequentWords(text, 1000000)
cl.saveObj(diction, "devdict")



rooms, meters, floors, locations, descriptions = cl.loadTestData()
text = ' '.join(descriptions)
diction = program.getNMostFrequentWords(text, 1000000)
cl.saveObj(diction, "testdict")