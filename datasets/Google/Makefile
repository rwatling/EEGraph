WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = web-Google.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/web-Google.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
