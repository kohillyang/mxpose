all:libheatpaf.so

libheatpaf.so : heatpaf.cpp
	g++ -c -fPIC heatpaf.cpp -o heatpaf.o
	g++ -shared -Wl,-soname,libheatpaf.so -o libheatpaf.so heatpaf.o

clean :
	-rm -vf libheatpaf.so heatpaf.o
m.PHONY : clean