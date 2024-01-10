
_make_lib:
	gcc -o ./test.o  -Wall -Werror -fpic -c test.c -lm
	gcc -shared -o ./libtest.so ./test.o -lm

test: _make_lib
	gcc -O2 -L./ -Wl,-rpath=./ main.c -o ./a.exe -lm -ltest
	./a.exe
	rm ./test.o ./libtest.so

rm:
	rm *.exe *.o *.so