#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

static void die(const char *msg) {
  int err = errno;
  fprintf(stderr, "[%d] %s\n", err, msg);
  std::abort();
}

int main() {

  // socket creation
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    die("socket()");
  }

  // connect
  struct sockaddr_in client_addr = {};
  client_addr.sin_family = AF_INET;
  client_addr.sin_port = htons(1234);                   // port
  client_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // IP 127.0.0.1

  int conn =
      connect(fd, (const struct sockaddr *)&client_addr, sizeof(client_addr));
  if (conn < 0) {
    die("connect()");
  }

  // write
  char msg[] = "hello";
  write(fd, msg, strlen(msg));

  // read
  char rbuf[64] = {};
  ssize_t n = read(fd, rbuf, sizeof(rbuf) - 1);
  if (n < 0) {
    die("read()");
  }

  // print
  printf("server says: %s\n", rbuf);
  close(fd);

  return 0;
}
