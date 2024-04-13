
#include "../include/buildroom.h"
#include "../include/constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  // set seed
  srand(time(NULL));

  // initialize room based on difficulty level
  int num_rooms = 10;
  enum LevelDifficulty difficulty = Medium;
  struct Room room = generateRooms(num_rooms, difficulty);
  printf("source: %d; destination: %d", room.source, room.destination);

  // solve maze using user input

  return 0;
}
