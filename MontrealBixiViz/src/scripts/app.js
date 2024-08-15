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

class BixiStation {
  constructor(
    capacity,
    eightd_has_key_dispenser,
    electric_bike_surchage_waiver,
    external_id,
    has_kiosk,
    is_charging,
    lat,
    lon,
    name,
    rental_methods,
    short_name,
    station_id
  ) {
    this.capacity = capacity;
    this.eightd_has_key_dispenser = eightd_has_key_dispenser;
    this.electric_bike_surchage_waiver = electric_bike_surchage_waiver;
    this.external_id = external_id;
    this.has_kiosk = has_kiosk;
    this.is_charging = is_charging;
    this.lat = lat;
    this.lon = lon;
    this.name = name;
    this.rental_methods = rental_methods;
    this.short_name = short_name;
    this.station_id = station_id;

    this.num_bikes_availables = 0;
    this.num_ebikes_available = 0;
    this.num_bikes_disabled = 0;
    this.num_docks_available = 0;
    this.num_docks_disabled = 0;
    this.is_installed = 0;
    this.is_renting = 0;
    this.is_returning = 0;
    this.last_reported = 0;
    this.eightd_has_available_keys = false;
    this.is_charging = false;
  }

  getInfo() {
    // TODO
  }
}

const translations = {
  en: {
    bixiStationStatusURL:
      "https://gbfs.velobixi.com/gbfs/en/station_status.json",
    bixiStationInformationURL:
      "https://gbfs.velobixi.com/gbfs/en/station_information.json",
    markerPopupHere: "You are here!",
  },
  fr: {
    bixiStationStatusURL:
      "https://gbfs.velobixi.com/gbfs/fr/station_status.json",
    bixiStationInformationURL:
      "https://gbfs.velobixi.com/gbfs/fr/station_information.json",
    markerPopupHere: "Vous êtes ici!",
  },
};

// initialize variables
var currentLanguage = "en";
// var popup = L.popup();
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map);
var bixiStationsArray = [];

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
  // Add a tile layer to the map
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution:
      '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  initBixiStationsOnMap();
}

async function fetchJSONData(url) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error with status ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching JSON data: ", error);
  }
}

async function initBixiStationsOnMap() {
  const data = await fetchJSONData(
    translations[currentLanguage].bixiStationInformationURL
  );

  bixiStationsArray = data.data.stations.map(
    (s) =>
      new BixiStation(
        s.capacity,
        s.eightd_has_key_dispenser,
        s.electric_bike_surchage_waiver,
        s.external_id,
        s.has_kiosk,
        s.is_charging,
        s.lat,
        s.lon,
        s.name,
        s.rental_methods,
        s.short_name,
        s.station_id
      )
  );

  updateBixiAvailability();
}

async function updateBixiAvailability() {
  // TODO
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
