# html-to-freq-2.py

import obo


text = ("chuj chuj chuj dupa kurwa cipa pizda dupa picza cwel chuj kurwa cwel ciec pizda dupa chuj kurwa dupa cipa").lower()
fullwordlist = obo.stripNonAlphaNum(text)
wordlist = obo.removeStopwords(fullwordlist, obo.stopwords)
dictionary = obo.wordListToFreqDict(wordlist)
sorteddict = obo.sortFreqDict(dictionary)

for s in sorteddict: print(str(s))