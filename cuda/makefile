.PHONY: src test clean

all: src test

# Change this.
src:
	cd src && $(MAKE) && cd ..

test:
	cd test && $(MAKE) && cd ..

clean:
	cd src && $(MAKE) clean && cd ..
	cd test && $(MAKE) clean && cd ..
