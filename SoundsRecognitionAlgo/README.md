# Sleep Tracker - Sounds Recognition Improvements

**Current state of the tracker**

In order to rate a user night of sleep, we are monitoring:
- sleep phases duration
- sounds recognition

**Challenges that we have**

- Sleep phases algorithm not accurate
- Sounds recognition not accurate
    * We are using YAMNET to recognize sounds and although our model report 
      having a confidence over 80%, user feedback report that sounds 
      recognition is only accurate 50% of the time
	+ [mixpanel](https://mixpanel.com/s/4tnR8B)
    * We suspect two elements at play:
	+ Discrepancy between the sound with which YAMNET was trained and 
	  the sounds that we are feeding the model.
	    + Solution E: hyperparameter tuning => Not possible with YAMNET
	    + Solution 1: train our own algorithm with 
		- Audioset dataset 
		- youtube dataset 
		- actual slips recording (needs to be annotated by us)
	+ With the tracker player, user are able to listen to sounds and guided 
	  listening during their nights, which adds noise during our recordings. 
	  We suspect sounds accuracy to have worsen because of that.
	    + Solution 1: Implement "Noise Cancellation" Model to remove music/sounds/guided 
	      content coming from our app


# Potential Projects to improve sleep tracker


Some projects around the sleep tracker:
- Sleep Phases Detection Accuracy
    * 
- Sleep Sounds Recognition
    * 
- Music/Guided Listening Sound Removal
    * 
- Sleep Insights
    * 
- Sleep Recordings


Challenges:
- Which project do we need to prioritize?
- We need to record full nights of sleep to improve our ML algorithms
    * How much time will it take to implement user opt-in? Will it affect adoption rate?
    * How will we store user sleeping information? 
	+ Sleep phases detection would need the whole night of sleep
	+ Sleep recording detection would only need the sounds clips


Goal of the project:
1. [ MIXPANEL ] Which categories are performing the worse according to users?
    a. [mixpanel](https://mixpanel.com/project/2481461/view/3023869/app/boards#id=3576568)
    b. False positive rate => which sounds are mislabeled: `edit_recording_confirm`



# The Sounds to Improve

**Bed Sheets**

- [YT Video - 1h](https://www.youtube.com/watch?v=7X0vXMLpi3U)
- Current Performance:

**Snoring**

- [YT Video]
- Current Performance:

**Farts**

- [YT Video - 10h](https://www.youtube.com/watch?v=rsRz2L-lUZQ)
- Current Performance:



# Resources

**Snoring Detection**

- [Hongwei Chen - Sleeping Monitor for Snoring Detection](https://tehunk.github.io/EECS395-mHealth/)

**Semi-Supervised Learning**

- [model says snoring, but it's actually fart => use that to retrain]


