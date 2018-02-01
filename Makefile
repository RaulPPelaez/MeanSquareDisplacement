all:
	$(MAKE) -C src/
	mkdir -p bin
	mv src/msd bin/

clean:
	rm -rf bin
