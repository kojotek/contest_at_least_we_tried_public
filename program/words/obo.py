# obo.py

stopwords = ['a', 'aby', 'ach', 'acz', 'aczkolwiek', 'aj', 'albo', 'ale', 'alez', 'az', 'bardziej', 'bardzo', 'beda', 'bedzie', 'bez', 'bo', 'bowiem', 'by', 'byc', 'byl', 'byla', 'byli', 'bylo', 'byly', 'bynajmniej', 'cala', 'cali', 'caly', 'ci', 'cie', 'ciebie', 'co', 'cokolwiek', 'cos', 'czasami', 'czasem', 'czemu', 'czy', 'czyli', 'daleko', 'dla', 'dlaczego', 'dlatego', 'do', 'dobrze', 'dokad', 'dosc', 'duzo', 'dwa', 'dwaj', 'dwie', 'dwoje', 'dzis', 'dzisiaj', 'gdy', 'gdyby', 'gdyz', 'gdzie', 'gdziekolwiek', 'gdzies', 'go', 'i', 'ich', 'ile', 'im', 'inna', 'inne', 'inny', 'innych', 'iz', 'ja', 'ja', 'jak', 'jakas', 'jakby', 'jaki', 'jakichs', 'jakie', 'jakis', 'jakiz', 'jakkolwiek', 'jako', 'jakos', 'je', 'jeden', 'jedna', 'jednak', 'jednakze', 'jedno', 'jego', 'jej', 'jemu', 'jesli', 'jest', 'jestem', 'jeszcze', 'jezeli', 'juz', 'kazdy', 'kiedy', 'kierunku', 'kilka', 'kims', 'kto', 'ktokolwiek', 'ktora', 'ktore', 'ktorego', 'ktorej', 'ktory', 'ktorych', 'ktorym', 'ktorzy', 'ktos', 'ku', 'lat', 'lecz', 'lub', 'ma', 'maja', 'mam', 'mi', 'miedzy', 'mimo', 'mna', 'mnie', 'moga', 'moi', 'moim', 'moj', 'moja', 'moje', 'moze', 'mozliwe', 'mozna', 'mu', 'musi', 'my', 'na', 'nad', 'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze', 'naszego', 'naszych', 'natomiast', 'natychmiast', 'nawet', 'nia', 'nic', 'nich', 'nie', 'niego', 'niej', 'niemu', 'nigdy', 'nim', 'nimi', 'niz', 'no', 'o', 'obok', 'od', 'okolo', 'on', 'ona', 'one', 'oni', 'ono', 'oraz', 'owszem', 'pan', 'pana', 'pani', 'po', 'pod', 'podczas', 'pomimo', 'ponad', 'poniewaz', 'powinien', 'powinna', 'powinni', 'powinno', 'poza', 'prawie', 'przeciez', 'przed', 'przede', 'przedtem', 'przez', 'przy', 'roku', 'rowniez', 'sa', 'sam', 'sama', 'sie', 'skad', 'soba', 'sobie', 'sposob', 'swoje', 'ta', 'tak', 'taka', 'taki', 'takie', 'takze', 'tam', 'te', 'tego', 'tej', 'ten', 'teraz', 'tez', 'to', 'toba', 'tobie', 'totez', 'totoba', 'trzeba', 'tu', 'tutaj', 'twoi', 'twoim', 'twoj', 'twoja', 'twoje', 'twym', 'ty', 'tych', 'tylko', 'tym', 'u', 'w', 'wam', 'wami', 'was', 'wasi', 'wasz', 'wasza', 'wasze', 'we', 'wedlug', 'wiec', 'wiecej', 'wiele', 'wielu', 'wlasnie', 'wszyscy', 'wszystkich', 'wszystkie', 'wszystkim', 'wszystko', 'wtedy', 'wy', 'z', 'za', 'zaden', 'zadna', 'zadne', 'zadnych', 'zapewne', 'zawsze', 'ze', 'zeby', 'zeznowu', 'znow', 'zostal']


def stripTags(pageContents):
    startLoc = pageContents.find("<p>")
    endLoc = pageContents.rfind("<br/>")

    pageContents = pageContents[startLoc:endLoc]

    inside = 0
    text = ''

    for char in pageContents:
        if char == '<':
            inside = 1
        elif (inside == 1 and char == '>'):
            inside = 0
        elif inside == 1:
            continue
        else:
            text += char

    return text

# Given a text string, remove all non-alphanumeric
# characters (using Unicode definition of alphanumeric).

def stripNonAlphaNum(text):
    import re
    return re.compile(r'\W+', re.UNICODE).split(text)
    
# Given a list of words, return a dictionary of
# word-frequency pairs.

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))
    
# Sort a dictionary of word-frequency pairs in
# order of descending frequency.

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

# Given a list of words, remove any that are
# in a list of stop words.

def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]

