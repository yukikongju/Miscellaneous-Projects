
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
void generateRooms(int num_rooms, enum LevelDifficulty levelDifficulty) {
  // determine start and end
  int source, destination;
  getSourceDestination(&source, &destination, num_rooms);

  // generate edges ie which rooms are connected - depending on level difficulty
  int graph[num_rooms][num_rooms];
  for (int i = 0; i < num_rooms; i++) {
    for (int j = 0; j < num_rooms; j++)
      graph[i][j] = 0;
  }

  switch (levelDifficulty) {
  case Easy:
    break;
  case Medium:
    generateNaiveEdges(num_rooms, source, destination, graph);
    break;
  case Hard:
    break;
  default:
    break;
  }

  // print graph + source/destination
  printf("source: %d; destination: %d \n", source, destination);
  printGraph(num_rooms, graph);
}
