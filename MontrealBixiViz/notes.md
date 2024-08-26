Steps:
- [X] Find the data
    - [X] Map of Montreal: leaflet
    - [X] [Map of Bixi](https://gbfs.velobixi.com/gbfs/gbfs.json)
    - [X] [Map of REV](https://donnees.montreal.ca/en/dataset/pistes-cyclables/resource/0dc6612a-be66-406b-b2d9-59c9e1c65ebf)
- [o] Move Marker
    - [X] Update user position on map click
    - [ ] Update Text Function to show current coordinates
    - [ ] Update user position based on address search
- [o] Toggle Language
    - [X] Switching language on button click
- [X] Add Arceaux
    - [X] Add arceaux to map
    - [X] Show distance from marker
    - [X] Add 'show/hide arceaux' button
- [o] Add Bixi
    - [X] init bixi json: information + availability
    - [X] Add bixi stations on the map
    - [X] Update Station Status with reload button
    - [X] Add pins for closest bixi that are disponible (green dispo; red not)
    - [X] Compute Distance from Marker to stations
    - [X] Change color based if user wants to drop or take a bixi: 'Looking for Bixi'; 'Putting back Bixi'
    - [ ] Find closest available stations and open popup on station click
    - [ ] Update Station Status every x minutes => `updateBixiAvailability()` + add last updated 
    - [ ] Cache station
    - [ ] Add user popup with nearest BIXI spot by walk
    - [ ] ~~Add pins for saved spot~~


Code Improvements:
- [X] Use same update/init function for language: `toggleLanguage` and `initButtonText`
- [X] No latence when fetching bixi station status in visual
- [X] Refractor bike circle update into class
- [X] Update distance from marker in Station class
    - [X] BixiStation
    - [X] ArceauxStation
- [X] Make "Station" class
    - [X] BixiStation extends Station
    - [X] ArceauxStation extends Station
- [X] Don't pass 'isVisible' when calling `updateStationVisual`
    - [X] clicking on Show Arceaux only change visibility
    - [X] add default visibility of station to true
    - [X] set bixi station visibility based on status load
- [X] File Refractoring
    - [X] Move translation into assets
    - [X] Move Coordinate class into models/
    - [X] Move 'Station' classes into models/
	- [X] set store variable for `currentLanguage` with [redux toolkit](https://redux.js.org/introduction/why-rtk-is-redux-today)
	- [X] move station and use store variable
    - [X] Fetch JSON Function into its own file

- [X] keep store variable for 'isLookingForBixi' and 'showArceaux' instead without updating visibility everytime ?

- [ ] Add button 'showREVPath'
    - [ ] load geojson file

Bug Fix:
- [X] Show Arceaux Button


