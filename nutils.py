# Autrhor: Nick Sebasco
# Date: 05/08/2021
# Version: 0
# Purpose:  Various helper functions used to manipulate data.
def findTrueKey(dic: dict) -> str: 
    """Given a dictionary as input, return the first key
    where a value of True is set as the value. An empty string is
    returned if no truth value was found. 
    """
    kv = [(key, dic[key]) for key in dic if dic[key]]
    return kv[0][0] if len(kv) > 0 else ""