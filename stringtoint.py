
def strToInt(exString):
    exIntArray = []
    stringLen = len(exString)
    zerosNeeded = [0] * (100 - stringLen)

    for c in exString:
        tempInt = ord(c)
        exIntArray.append(tempInt)

    exIntArray = exIntArray + zerosNeeded

    return exIntArray