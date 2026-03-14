#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void die(const char *msg) { printf("%s\n", msg); }

int main() {
  int fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    die("socket() error");
    return 1;
  }

  struct sockaddr_in addr_in = {};
  addr_in.sin_family = PF_INET;
  addr_in.sin_len = sizeof(addr_in);
  addr_in.sin_port = htons(1234);
  if (connect(fd, (const struct sockaddr *)&addr_in, sizeof(addr_in)) < 0) {
    die("connect() error");
  }

  // write to server
  const char *msg = "hello";
  write(fd, msg, strlen(msg));

  // read response from server
  char buf[64] = {};
  read(fd, buf, sizeof(buf));
  printf("%s\n", buf);

  // close
  close(fd);
  return 0;
}
