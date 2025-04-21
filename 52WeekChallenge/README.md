# 52 Week Challenge

One project a week for a whole year!

## ----- The Projects -----

1. [ ] [Server](#server)
2. [ ] [AUDL](#audl)
3. [ ] [Plant Watering System](#plant-watering-system)
4. [ ] [Room Movement](#room-movement)
5. [ ] [Remote Control Car](#remote-control-car)
6. [ ] [Dinosaur Bot](#dinosaur-bot)
7. [ ] [WhoSaidIt Tweet](#whosaidit-tweet)
8. [ ] [Noise Cancellation](#noise-cancellation)
9. [ ] [Residual Image](#residual-image)
10. [ ] [Multiplayer Maze Battle](#multiplayer-maze-battle)
11. [ ] [Audio Image](#audio-image)
12. [ ] [Toralizer](#toralizer)
13. [ ] [Dominion-like](#dominion-like)
14. [ ] [Twich Stream](#twich-stream)
15. [ ] [Fractal Generation](#fractal-generation)
16. [ ] [Terrain Generator](#terrain-generator)
17. [ ] [vim-wiki calendar template](#vim-wiki-calendar-template)
18. [ ] [Nostale Bot](#nostale-bot)
19. [ ]
20. [ ]
21. [ ]
22. [ ]
23. [ ]
24. [ ]
25. [ ]
26. [ ]
27. [ ] [Drone](#drone)

# ----- Projects Details -----

## Server

- Tags: [Hardware, Network]
- Idea: Turn an old computer into a server to store movies, songs, run jobs
- Goal: ssh to server to train model, host postgres databases for future projects,
  watch/store movies/music

#### AUDL

- Tags: [DE/DS/DL]
- Idea: Create a pipeline to fetch AUDL data and use that data for
    * train RL algorithm to find optimal strategy for each team
    * host data visualization website + social network analysis
- Technologies:
    * Databases:
	+ MongoDB (JSON for game events),
	+ PosgreSQL (metadata: players, games, teams)
	+ Neo4J (network analysis)
	+ Redis (cache server data)
    * Airflow, dbt,


#### Plant Watering System

- Tags: [IOT]
- Idea: Build an arduino sensor that measure soil humidity and water one plant
- Goal: Learn IOT, realtime system,
- Technologies:
    * Python/C++
    * Kafka (ingest data in realtime)
    * Grafana (track and visualize) -> OTEL-LGTM Stack + OpenTelemetry
    * OpenTelemetry (observability instrumentations for metrics, traces, logs)

#### Room Movement

- Tags: [IOT/DE/DevOps]
- Idea: Monitor the amount of time I stay in my room and generate analytics
- Goal: Leverage computer vision
- Technologies:
    * OpenCV
    * Grafana

#### Remote Control Car

- Tags: [IOT]
- Idea:
- Goal:
- Technologies:


#### Dinosaur Bot

- Tags: [IOT]
- Idea: Build a arduino sensor that plays Dinosaur Offline Game
- Goal: Read data from sensor to activate arm
- Technologies:
    *


#### WhoSaidIt Tweet

- Tags: [DL/SE]
- Idea: Guess if the tweet was computer generated or a real tweet from X celebrity
- Goal: Use style transfer learning and build a front-end app for it, store
  user score in database
    1. Get tweets from popular people
    2. Train Model
    3. Generate fake tweets and store in databases
    4. Build Mobile/Web App
- Technologies:
    * DL: PyTorch, MLFlow
    * Databases: PosgreSQL, Cassandra
    * SE: iOS & Javascript (MERN)

#### Noise Cancellation

- Tags: [DL / Signal Processing]
- Idea: Given an audio file, remove any noise
- Goal: use sound difference, anti-aliasing, U-Net to clean up noise. Usage: NOAA
- Technologies:


#### Residual Image

- Tags: [Maths/CV/SE]
- Idea: Given an image, generate a function (Lagrangian) and a dataset whose residual creates the image
- Goal:
- Technologies:
    * Vanilla Javascript

#### Multiplayer Maze Battle

- Tags: [Game/Maths/SE/Networking]
- Idea: Make a multiplayer maze game with the following modes:
    * Race Mode: players race to escape first
    * Co-op Mode: solve puzzle together to unlock areas
    * Battle Mode: players can drop traps, block paths or steal boost
- Goal:
    * Learn how networking works for in-real time game
    * Learn the algorithms to generate maze
- Technologies:
    * React/C++

#### Audio Image

- Tags: [Computer Vision / Signal Processing / SE]
- Idea: Given an object/animal, generate a soundwave that looks like it and sounds like it (ex: cow, car, ...)
- Goal:
    * Learn how soundwaves work
- Technologies:

#### Toralizer

- Tags:
- Idea:
- Goal:
- Technologies:

[Network] **Toralizer** =>

#### Dominion-like

- Tags: [Game Dev / Network / RL]
- Idea: Card building game with 8-bit design; include the following mechanics: ; game mode: minus points for every 5 cards in hand; draw a bronze every turn; generate using AI
- Goal:
    * Use RL to build bot
- Technologies:

#### Twich Stream

- Tags:
- Idea:
- Goal:
    * Learn video upload
    * Learn video reads: how packets are being sent over the network
- Technologies:

#### Fractal Generation

- Tags: [Maths]
- Idea: Mandelbrot, Julia Sets
- Goal:
- Technologies:

#### Terrain Generator

- Tags: [Game/Graphics/DL]
- Idea: Generate random islands, caves or cities with Perlin/Simplex noise
- Goal:
- Technologies:


#### vim-wiki calendar template

- Tags: [Plugin]
- Idea:
- Goal:
- Technologies:

#### Nostale Bot

- Tags:
- Idea:
- Goal:
- Technologies:

[Game/ML] **Nostale Bot** =>

#### Drone

- Tags: [IOT]
- Idea:
- Goal:
- Technologies:

# ----- IDEAS -----

## Data Engineering & Databases

## Computer Vision

- self-driving car

## Natural Language Processing

## Networks

- [inc] Web Server => Http Server is C
- [inc] Networked Multiplayer Game =>
- [inc] Blockchain =>
- [inc] Trading System w/ threads =>

## Compilers

- [inc] Tiny Compiler =>
- [inc] AEDIROUM Programming Language =>
- [inc] Operating System Kernel Module => Linux kernel module

## Maths

## Game

- Game Engine

## Computer Graphics

- Navier-Stokes to simulate water flow
- Particle based smoke and fire
- Raycasting

## Other
