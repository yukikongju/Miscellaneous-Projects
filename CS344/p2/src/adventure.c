
#include "../include/buildroom.h"
#include "../include/constants.h"
#include <stdio.h>

int main() {

  int num_rooms = 10;
  enum LevelDifficulty difficulty = Easy;
  generateRooms(num_rooms, difficulty);

  return 0;
}
