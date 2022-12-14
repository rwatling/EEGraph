WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = roadNet-CA.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/roadNet-CA.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
