cc=g++
cflags=-Wall -g
exe=main
obj=main.o ID3.o C4d5.o
$(exe):$(obj)
	$(cc) $(cflags) -o $(exe) $(obj)
main.o: main.cpp ID3.h C4d5.h
	$(cc) $(cflags) -c main.cpp
C4d5.o: C4d5.cpp C4d5.h ID3.o
	$(cc) $(cflags) -c C4d5.cpp 
ID3.o: ID3.cpp ID3.h
	$(cc) $(cflags) -c ID3.cpp
.PHONY: clean print
clean:
	rm -rf *.o
print: *.cpp
	lpr -p $?
	touch print
