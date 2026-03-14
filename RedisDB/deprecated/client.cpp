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
#include <unistd.h>
#include <vector>

using std::vector;

// const size_t k_max_msg = 4096;
const size_t k_max_msg = 32 << 20;

static void die(const char *msg) {
  int err = errno;
  fprintf(stderr, "[%d] %s\n", err, msg);
  std::abort();
}

static void buf_append(std::vector<uint8_t> &buf, const uint8_t *data,
                       size_t len) {
  buf.insert(buf.end(), data, data + len);
}

static void msg(const char *msg) { fprintf(stderr, "%s\n", msg); }

static int32_t write_all(int fd, const uint8_t *buf, size_t n) {
  while (n > 0) {
    ssize_t rv = write(fd, buf, n);
    if (rv < 0) {
      return -1;
    }
    n -= (size_t)rv;
    buf += rv;
  }
  return 0;
}

static int32_t read_full(int fd, uint8_t *buf, size_t n) {
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

static int32_t send_requests(int fd, const uint8_t *data, size_t len) {
  if (len > k_max_msg) {
    return -1;
  }
  // std::vector<uint8_t> wbuf[4 + k_max_msg]; // if fixed raw byte array
  std::vector<uint8_t> wbuf;
  buf_append(wbuf, (const uint8_t *)&len, 4);
  buf_append(wbuf, data, len);
  // memcpy(wbuf, &len, 4);
  // memcpy(&wbuf[4], text, len);
  if (int32_t err = write_all(fd, wbuf.data(), wbuf.size())) {
    return err;
  }

  return 0;
}

static int32_t read_results(int fd) {
  // char rbuf[4 + k_max_msg];
  std::vector<uint8_t> rbuf;
  rbuf.resize(4);
  errno = 0;
  int32_t err = read_full(fd, &rbuf[0], 4);
  if (err) {
    msg(errno == 0 ? "EOF" : "read() error");
  }

  uint32_t len = 0;
  memcpy(&len, rbuf.data(), 4);
  if (len > k_max_msg) {
    msg("server message too long");
    return -1;
  }
  err = read_full(fd, &rbuf[4], len);
  if (err) {
    msg("read() error");
    return err;
  }

  // do something
  printf("server: %.*s\n", len, &rbuf[4]);
  return 0;
}

// int32_t query(int fd, const char *text) {
//// check if message is not too long
//// size_t len = strlen(msg); // why not size_t ?
// uint32_t len = (uint32_t)strlen(text);
// if (len > k_max_msg) {
// return -1;
//}

//// send request
// char wbuf[4 + k_max_msg];
// memcpy(wbuf, &len, 4);
// memcpy(&wbuf[4], text, len);
// if (int32_t err = write_all(fd, wbuf, 4 + len)) {
// return err;
//}

//// read reply: (1) check header (2) check length (3) get body
// char rbuf[4 + k_max_msg];
// errno = 0;
// int32_t err = read_full(fd, rbuf, 4);
// if (err) {
// msg(errno == 0 ? "EOF" : "read() error");
//}
// memcpy(&len, rbuf, 4);
// if (len > k_max_msg) {
// msg("server message too long");
// return -1;
//}
// err = read_full(fd, &rbuf[4], len);
// if (err) {
// msg("read() error");
// return err;
//}

// do something
// printf("server: %.*s\n", len, &rbuf[4]);
// return 0;
//}

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
  vector<std::string> query_list = {"hello1", "hello2", "hello3"};
  for (const std::string &s : query_list) {
    int32_t err = send_requests(fd, (uint8_t *)s.data(), s.size());
    if (err) {
      goto L_DONE;
    }
  }

  // read response from server
  for (size_t i = 0; i < query_list.size(); i++) {
    int32_t err = read_results(fd);
    if (err) {
      goto L_DONE;
    }
  }

L_DONE:
  close(fd);
  return 0;
}
