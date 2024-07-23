#ifndef BUILDROOM_H
#define BUILDROOM_H

#include "constants.h"

#define MAX_VERTICES 20

struct Room {
  int source;
  int destination;
  int *graph;
  // int graph[MAX_VERTICES][MAX_VERTICES];
};

struct Room generateRooms(int num_rooms, enum LevelDifficulty);

#endif
