main:main.o
	g++ main.o `pkg-config --libs opencv` -o main
main.o:main.cpp
	gcc `pkg-config --cflags opencv` -g -c main.cpp
