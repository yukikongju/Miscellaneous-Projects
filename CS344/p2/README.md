# Adventure

[link](https://github.com/TylerC10/CS344/tree/master/Project%202%20-%20Adventure)

[Colossal Cave Adventure](https://en.wikipedia.org/wiki/Colossal_Cave_Adventure)

**How to play**

Input 1-2 letters to give direction

Moves: [ 8 directions ]
- n: north
- ne: north-east
- nw: north-west
- s: south
- se: south-east
- sw: south-west
- w: west
- e: east

**The Game**

Player tries to escape a maze by trying to open different doors. Every time 
a door is opened, the player needs to wait a random amount of time before 
exploring the next room. The level finishes when the player finds the exit

Game mode:
- Levels: unlimited amount of rooms
- Survival: try to find the exit in the given amount of time

Difficulty:
- Easy: map of all visited room 
- Medium: only visited rooms are shown (blind)
- Hard: no map given

**Generating the maze**

1. The maze will be a *graph*
    a. Generate n rooms
    b. Generate edges between each rooms
2. Summon player in a random room
3. When a player opens a room, start a thread. Unlock the door when the thread 
   is done

**Implementing the map**


# Resources

**Multi-threading**
