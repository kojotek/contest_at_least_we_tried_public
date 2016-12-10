# program.py

from words import obo

def getNMostFrequentWords(text, n):
	txt = text.lower()
	fullwordlist = obo.stripNonAlphaNum(txt)
	wordlist = obo.removeStopwords(fullwordlist, obo.stopwords)
	dictionary = obo.wordListToFreqDict(wordlist)
	diction = obo.sortFreqDict(dictionary)
	#diction = {s[0]: s[1] for s in diction}
	#invdict = {s[1]: s[0] for s in sorteddict}
	return diction[:n]