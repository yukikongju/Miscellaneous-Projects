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
    - [ ] Move translation into assets
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
- [ ] Use same update/init function for language: `toggleLanguage` and `initButtonText`



