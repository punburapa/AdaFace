# download_pairs.py
import urllib.request
urllib.request.urlretrieve(
    "http://vis-www.cs.umass.edu/lfw/pairs.txt",
    "data/pairs.txt"
)