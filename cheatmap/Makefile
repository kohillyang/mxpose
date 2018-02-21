
libheatmaps.so : heatmaps.o
	gcc -shared -Wl,-soname,libheatmaps.so -o libheatmaps.so heatmaps.o

heatmaps.o : heatmaps.c
	gcc -c -fPIC heatmaps.c -o heatmaps.o

clean :
	-rm -vf libheatmaps.so heatmaps.o heatmaps.pyc
m.PHONY : clean
	