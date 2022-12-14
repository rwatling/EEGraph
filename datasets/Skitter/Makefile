WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = as-skitter.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/as-skitter.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
