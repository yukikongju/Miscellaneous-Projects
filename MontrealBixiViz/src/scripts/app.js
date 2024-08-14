class Coordinate {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
}

// Step 1: initialize Montreal coordinates
const coord = new Coordinate(45.5335, -73.6483);

// Step 2: Initialize the map
var map = L.map("map").setView([coord.x, coord.y], 13);

// Step 3: Add a tile layer to the map
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution:
    '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

// Step 4: Optionally add a marker
L.marker([coord.x, coord.y]).addTo(map).bindPopup("Montreal, QC").openPopup();
