
#include "../include/buildroom.h"
#include "../include/constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int getUserInput() {
  int userInput;
  printf("Room to move to:");
  scanf("%d", &userInput);
  return userInput;
}

int main() {
  // set seed
  srand(time(NULL));

  // initialize room based on difficulty level
  int num_rooms = 10;
  enum LevelDifficulty difficulty = Medium;
  struct Room room = generateRooms(num_rooms, difficulty);
  printf("source: %d; destination: %d", room.source, room.destination);

  // solve maze using user input
  bool isRoomSolved = false;
  int currentRoom = room.source;
  while (!isRoomSolved) {
    int userInput = getUserInput();

    // move to room if possible
    if (userInput >= num_rooms) {
      printf("Please enter a room between 0 and %d\n", num_rooms - 1);
    } else if (room.graph[currentRoom][userInput] == 0) {
      printf("Room %d is unaccessible from room %d\n", userInput, currentRoom);
    } else if (room.graph[currentRoom][userInput] == 3) {
      printf("You reached the destination!");
      isRoomSolved = true;
    } else {
      currentRoom = userInput;
    }
  }

  return 0;
}
