compile:
	g++ -std=c++11 -Wall -Wextra -Og -g server.cpp -o server
	g++ -std=c++11 -Wall -Wextra -Og -g client.cpp -o client
	# in terminal 1: ./server
	# in terminal 2: ./client
	# strace ./server >/dev/null

clean:
	rm -rf client.dSYM/ server.dSYM/
	rm server client
	# rm server.log client.log
