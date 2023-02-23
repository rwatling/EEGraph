WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = wiki-Talk.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/wiki-Talk.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
