WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = com-youtube.ungraph.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
