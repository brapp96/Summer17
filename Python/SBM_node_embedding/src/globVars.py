# settings.py
"""
Contains globals for the VEC suite.
"""

def init():
    """
    Initializes global variables
    """
    global FILEPATH
    global DEBUG
    FILEPATH = '' # dummy initialization - unsure if necessary or not
    DEBUG = 0

def printDebug(string):
    "Print a string only if DEBUG flag is set"
    if DEBUG:
        logfile = open(FILEPATH+'/test.log', 'a')
        logfile.write(string+'\n')
        logfile.close()     
