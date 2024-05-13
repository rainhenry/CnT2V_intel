
touchserver:main.o xdo.o
	gcc -g -o touchserver main.o xdo.o -lX11 -lXtst -lXinerama -lxkbcommon

main.o:main.c
	gcc -g -c -o main.o main.c

xdo.o:xdo.c
	gcc -g -c -o xdo.o xdo.c

clean:
	-rm -rf *.o
	-rm -rf touchserver

