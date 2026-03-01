#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

static void die(const char *msg) {
  int err = errno;
  fprintf(stderr, "[%d] %s\n", err, msg);
  std::abort();
}

void process_connection(int connfd) {
  // read
  char rbuf[64] = {};
  ssize_t n =
      read(connfd, rbuf, sizeof(rbuf) - 1); // why doesn't work with strlen()
  if (n < 0) {
    die("read()");
  }

  printf("client: %s\n", rbuf);

  // write
  char wbuf[] = "received";
  write(connfd, wbuf, strlen(wbuf));
}

int main() {
  // create socket
  int fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    die("socket()");
  }

  // bind socket address
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(1234);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int rv = bind(fd, (const struct sockaddr *)&addr, sizeof(addr));
  if (rv < 0) {
    die("bind()");
  }

  // listen to endpoint
  rv = listen(fd, SOMAXCONN);
  if (rv < 0) {
    die("listen()");
  }

  // accept connection
  while (true) {
    struct sockaddr_in conn_addr = {};
    socklen_t socklen_addr = sizeof(conn_addr);
    int connfd = accept(fd, (struct sockaddr *)&conn_addr, &socklen_addr);
    if (connfd < 0) {
      continue; // invalid connection
    }

    // process the connection
    process_connection(connfd);
    close(connfd);
  }

  return 0;
}
