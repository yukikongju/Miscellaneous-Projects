
#include "../include/buildroom.h"
#include "../include/constants.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 100

void getSourceDestination(int *source, int *destination, int num_rooms) {
  *source = rand() % num_rooms;
  do {
    *destination = rand() % num_rooms;
  } while (*source == *destination);
}

void printGraph(int num_rooms, int (*graph)[num_rooms]) {
  for (int i = 0; i < num_rooms; i++) {
    for (int j = 0; j < num_rooms; j++) {
      printf("%d ", graph[i][j]);
    }
    printf("\n");
  }
}

void generateNaiveEdges(int num_rooms, int source, int destination,
                        int (*graph)[num_rooms]) {
  // -- generate path from source to destination naively
  int current = source;
  while (current != destination) {
    int next = rand() % num_rooms;
    graph[current][next] = 1;
    current = next;
  }
}

// Generating random rooms as a graph
// 0: no edge; 1: has edge; 2: source; 3: destination
struct Room generateRooms(int num_rooms, enum LevelDifficulty levelDifficulty) {
  // determine start and end
  struct Room room;
  room.source = -1;
  room.destination = -1;
  /* int source, destination; */
  getSourceDestination(&room.source, &room.destination, num_rooms);

  // generate edges ie which rooms are connected - depending on level difficulty
  int graph[num_rooms][num_rooms];
  for (int i = 0; i < num_rooms; i++) {
    for (int j = 0; j < num_rooms; j++)
      graph[i][j] = 0;
  }
  graph[room.source][room.source] = 2;
  graph[room.destination][room.destination] = 3;

  switch (levelDifficulty) {
  case Easy:
    break;
  case Medium:
    generateNaiveEdges(num_rooms, room.source, room.destination, graph);
    break;
  case Hard:
    break;
  default:
    break;
  }
  room.graph = *graph;

  // print graph + source/destination
  /* printf("source: %d; destination: %d \n", room.source, room.destination); */
  /* printGraph(num_rooms, graph); */

  // return room struct
  return room;
}
