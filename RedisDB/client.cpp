#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

const size_t k_max_msg = 4096;

static void die(const char *msg) {
  int err = errno;
  fprintf(stderr, "[%d] %s\n", err, msg);
  std::abort();
}

static void msg(const char *msg) { fprintf(stderr, "%s\n", msg); }

static int32_t write_all(int fd, char *buf, size_t n) {
  while (n > 0) {
    int rv = write(fd, buf, n);
    if (rv < 0) {
      return -1;
    }
    n -= (size_t)rv;
    buf += rv;
  }
  return 0;
}

static int32_t read_full(int fd, char *buf, size_t n) {
  while (n > 0) {
    int rv = read(fd, buf, n);
    if (rv < 0) {
      return -1;
    }
    n -= (size_t)rv;
    buf += rv;
  }
  return 0;
}

int32_t query(int fd, const char *text) {
  // check if message is not too long
  // size_t len = strlen(msg); // why not size_t ?
  uint32_t len = (uint32_t)strlen(text);
  if (len > k_max_msg) {
    return -1;
  }

  // send request
  char wbuf[4 + k_max_msg];
  memcpy(wbuf, &len, 4);
  memcpy(&wbuf[4], text, len);
  if (int32_t err = write_all(fd, wbuf, 4 + len)) {
    return err;
  }

  // read reply: (1) check header (2) check length (3) get body
  char rbuf[4 + k_max_msg];
  errno = 0;
  int32_t err = read_full(fd, rbuf, 4);
  if (err) {
    msg(errno == 0 ? "EOF" : "read() error");
  }
  memcpy(&len, rbuf, 4);
  if (len > k_max_msg) {
    msg("server message too long");
  }
  err = read_full(fd, &rbuf[4], len);
  if (err) {
    msg("read() error");
  }

  // do something
  printf("server: %.*s\n", len, &rbuf[4]);
  return 0;
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

  // send requests to server
  int32_t err = query(fd, "hello");
  if (err) {
    goto L_DONE;
  }

  // sending another requests to server
  err = query(fd, "hello2");
  if (err) {
    goto L_DONE;
  }

  // write
  // char msg[] = "hello";
  // write(fd, msg, strlen(msg));

  // // read
  // char rbuf[64] = {};
  // ssize_t n = read(fd, rbuf, sizeof(rbuf) - 1);
  // if (n < 0) {
  //   die("read()");
  // }

  // // print
  // printf("server says: %s\n", rbuf);

L_DONE:
  close(fd);
  return 0;
}
