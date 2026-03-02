#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

const size_t k_max_msg = 4096;

static void die(const char *msg) {
  int err = errno;
  fprintf(stderr, "[%d] %s\n", err, msg);
  std::abort();
}

static void msg(const char *msg) { fprintf(stderr, "%s\n", msg); }

static int32_t read_full(int fd, char *buf, size_t n) {
  while (n > 0) {
    size_t rv = read(fd, buf, n);
    if (rv < 0) {
      return -1;
    }
    // assert((size_t)rv <= n);
    n -= (size_t)rv;
    buf += rv;
  }
  return 0;
}

static int32_t write_all(int fd, char *buf, size_t n) {
  while (n > 0) {
    size_t rv = write(fd, buf, n);
    if (rv < 0) {
      return -1;
    }
    // assert((size_t)rv <= n);
    n -= (size_t)rv;
    buf += rv;
  }
  return 0;
}

static int32_t process_request(int connfd) {
  // verify if message is correct - header is 4 bytes
  char rbuf[4 + k_max_msg];
  errno = 0;
  int err = read_full(connfd, rbuf, 4);
  if (err < 0) {
    msg(errno == 0 ? "EOF" : "read() error");
  }

  // read
  uint32_t len = 0;
  memcpy(&len, rbuf, 4);
  if (len > k_max_msg) {
    msg("client message too long");
    return -1;
  }
  err = read_full(connfd, &rbuf[4], len);
  if (err < 0) {
    msg("read() error");
  }

  // write
  printf("client says %.*s\n", len, &rbuf[4]);

  // respond using protocol
  const char reply[] = "received";
  char wbuf[4 + sizeof(reply)];
  len = (uint32_t)strlen(reply);
  memcpy(wbuf, &len, 4);
  memcpy(&wbuf[4], reply, len);
  return write_all(connfd, wbuf, 4 + len);
}

void process_connection(int connfd) {
  // read
  char rbuf[64] = {};
  // note: doesn't work with strelen() because rbuf empty
  ssize_t n = read(connfd, rbuf, sizeof(rbuf) - 1);
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
    // process_connection(connfd); // naive
    while (true) {
      int err = process_request(connfd);
      if (err) {
        break;
      }
    }
    close(connfd);
  }

  return 0;
}
