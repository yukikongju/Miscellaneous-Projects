class Coordinate {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  updatePosition(x, y) {
    this.x = x;
    this.y = y;
  }
}

// initialize variables
var popup = L.popup();
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map).bindPopup("You are here"); // .openPopup();

function loadMap() {
  // Step 3: Add a tile layer to the map
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution:
      '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
}

function updateMarkerPosition(e) {
  coord.updatePosition(e.latlng.lat, e.latlng.lng);

  // TODO: set popup content to see closest Bixi stations
  var popupContent = "Clicked here!";
  marker.setLatLng(e.latlng).setPopupContent(popupContent).openOn(map);
}

function onMapClick(e) {
  updateMarkerPosition(e);
}

function main() {
  loadMap();
  map.on("click", onMapClick);
}

main();
