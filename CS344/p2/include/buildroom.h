#ifndef BUILDROOM_H
#define BUILDROOM_H

#include "constants.h"

struct Room {
  int source;
  int destination;
  int *graph;
};

struct Room generateRooms(int num_rooms, enum LevelDifficulty);

#endif
