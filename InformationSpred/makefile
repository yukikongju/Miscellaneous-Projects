CXX = g++
output_bin = a.out

compile:
	$(CXX) -I./include src/*.cpp -o $(output_bin) -std=c++11
	./$(output_bin)

init:
	bear make
	ctags -R .

