#include <fcntl.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/socket.h>

// INC
//
/* Sys Calls: socket, bind, listen, accept, recv, open, sendfile */
/* https://www.youtube.com/watch?v=2HrYIl6GpYg&list=PL0tgH22U2S3Giz-yIxVQTEKIZpMBP0tMf&index=1&ab_channel=NirLichtman
 */

// Q: what is a file descriptor?

void main() {
  int s = socket(PF_INET6, SOCK_STREAM, 0);
  struct sockaddr_in addr = {PF_INET6, 0x09f1, 0}; // port 8080

  bind(s, &addr, sizeof(addr));

  listen(s, 10); // backlog: num of connections that can be queued

  int client_fd = accept(s, 0, 0);
  char buffer[256] = {0};
  recv(client_fd, buffer, 256, 0);

  // GET /file.html

  char *f = buffer + 5;
  *strchr(f, ' ') = 0;

  opened_fd = open(f, O_RDONLY);
}
