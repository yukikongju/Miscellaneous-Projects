# ML Projects Ideas

**Guided Content**

- [o] "Because you listened to" Carousel / "Radio"
    * What: Suggest guided content similar to the one the users has listened
      in the past
    * How: Content Filtering based on Doc2Vec and KNN
    * Caveats: Don't recommend content already listened
- "Mixes for you" Carousel
    * What: Create new mixes for you based on sounds a user has listened to
    * How: Look at user mixes from listening events (can be found in `sounds`
      event param) + generate custom name for it with ChatGPT.
    * Caveats: Filter the non-custom one. When computing, only consider
      the mixes made organically, not the one suggested by carousel

**Mixer/Player**

- "Pairs well with" Carousel in Mixer/Player
    * What: Recommend sounds/mixes that pair well with guided content in player;
      with music/sounds/mixes in mixer
    * How: Look at sounds/mixes users put with each guided content
    * Other usages:
	+ When selecting guided content, users can add generated mixes


**Tracker**

- Sound Denoising

**Miscellaneous**

- Spotify Wrapped
    * What: Show the user their metrics in the last year => guided content
      listened, number of minutes listened, mixes created, ...
