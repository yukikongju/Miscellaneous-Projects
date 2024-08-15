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

const translations = {
  en: {
    markerPopupHere: "You are here!",
  },
  fr: {
    markerPopupHere: "Vous êtes ici!",
  },
};

// initialize variables
var currentLanguage = "en";
// var popup = L.popup();
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map);

function toggleLanguage() {
  currentLanguage = currentLanguage === "en" ? "fr" : "en";
  button = document.getElementById("language-button");
  button.textContent = currentLanguage;

  updateText();
}

function updateText() {
  // update marker popup
  marker.setPopupContent(translations[currentLanguage].markerPopupHere);
}

function initLanguageText() {
  // init language button
  button = document.getElementById("language-button");
  button.textContent = currentLanguage;

  // init Marker popup
  marker.bindPopup(translations[currentLanguage].markerPopupHere);
}

function initMap() {
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
  // marker.setLatLng(e.latlng).setPopupContent(popupContent).openOn(map);
  marker.setLatLng(e.latlng).openOn(map);
  updateText();
}

function onMapClick(e) {
  updateMarkerPosition(e);
}

function main() {
  initLanguageText();
  initMap();

  // add listening events
  map.on("click", onMapClick);
}

main();
