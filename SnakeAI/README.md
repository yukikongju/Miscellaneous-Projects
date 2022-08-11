# SnakeAI

Play against a snake AI



# The Agent

**Defining the reward**

- eat food: +20
- move: -1
- game over: -50

**Defining the action**

- straight: [1,0,0]
- turn right: [0,1,0]
- turn left: [0,0,1]

**Defining the States**

- Danger Straight, right, left
- Direction left, right, up, down
- food left, right, up, down

**Neural Network Parameters**

- input: states (x11)
- output: action (x3)

# Ressources

- [Teach AI to play Snake - Python Engineer](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)


