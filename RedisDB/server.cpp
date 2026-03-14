#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/_endian.h>
#include <sys/socket.h>
#include <unistd.h>

void die(const char *msg) { printf("%s\n", msg); }

void process_request(int connfd) {
  // read client request
  char buf[64] = {};
  read(connfd, buf, sizeof(buf));
  printf("%s\n", buf);

  // respond to client
  const char *msg = "world";
  write(connfd, msg, strlen(msg));
}

int main() {
  int fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    die("socket() error");
    return 1;
  }

  // bind the socket to an address and port to avoid "address already in use"
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt,
                 sizeof(opt)) < 0) { // SO_REUSEADDR
    die("setsockopt() error");
    return 1;
  }

  // https://stackoverflow.com/questions/21099041/why-do-we-cast-sockaddr-in-to-sockaddr-when-calling-bind
  struct sockaddr_in addr_in = {};
  addr_in.sin_family = AF_INET;
  addr_in.sin_len = sizeof(addr_in);
  addr_in.sin_port = htons(1234);

  if (bind(fd, (const struct sockaddr *)&addr_in, sizeof(addr_in)) < 0) {
    die("bind() error");
    return 1;
  }

  if (listen(fd, 5) < 0) {
    die("listen() error");
    return 1;
  }

  while (true) {
    struct sockaddr_in client_addr = {};
    socklen_t client_len = sizeof(client_addr);
    int connfd = accept(fd, (struct sockaddr *)&client_addr, &client_len);

    // read write to connection
    printf("Connected from port %d\n", ntohs(client_addr.sin_port));
    process_request(connfd);

    close(connfd);
  }

  return 0;
}
