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

Moves: select room number


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
	+ Easy: 
	    1. Create Straight forward path: start from source and advance until 
	       we find destination
	    2. With remaining rooms, link them together and link them back to 
	       one of the nodes in the path. must not be source nor destination
	+ Medium: 
	    1. 
	+ Hard:
	    1. 
2. Summon player in a random room
3. When a player opens a room, start a thread. Unlock the door when the thread 
   is done

**Implementing the map**

- Graph will be an *adjacency graph*
- Types of rooms:
    * Simple Room: once the source, destination and graph have been generated, 
      they don't change
    * Open room: the user can open some room, and an edge will be created 
      after a certain amount of time.

**TODOS**

- [ ] Shortest path algorithm for adjacency graph
- [ ] Compute score based on (1) time to solve (2) number of moves compared to 
      shortest path

# Resources

**Multi-threading**
