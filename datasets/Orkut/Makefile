WGET = wget
UNZIP = tar -xf
GRAPH_ZIP_FILE = download.tsv.orkut-groupmemberships.tar.bz2
GRAPH_URL = http://konect.cc/files/download.tsv.orkut-groupmemberships.tar.bz2

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)
