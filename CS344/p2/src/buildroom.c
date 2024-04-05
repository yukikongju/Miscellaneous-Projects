
#include "../include/constants.h"
#include <stdio.h>
#include <stdlib.h>

// Generating random rooms as a graph
void generateRooms(int num_rooms, enum LevelDifficulty levelDifficulty) {
  // determine start and end
  int start = rand() % num_rooms;
  int end;
  do {
    end = rand() % num_rooms;
  } while (start == end);
  printf("start: %d ; end: %d", start, end);

  // generate edges ie which rooms are connected
  int rooms[num_rooms];
}
