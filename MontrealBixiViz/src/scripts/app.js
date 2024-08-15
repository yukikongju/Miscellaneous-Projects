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

    this.num_bikes_available = 0;
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

  updateAvailability(
    num_bikes_available,
    num_ebikes_available,
    num_bikes_disabled,
    num_docks_available,
    num_docks_disabled,
    is_installed,
    is_renting,
    is_returning,
    last_reported,
    eightd_has_available_keys,
    is_charging
  ) {
    this.num_bikes_available = num_bikes_available;
    this.num_ebikes_available = num_ebikes_available;
    this.num_bikes_disabled = num_bikes_disabled;
    this.num_docks_available = num_docks_available;
    this.num_docks_disabled = num_docks_disabled;
    this.is_installed = is_installed;
    this.is_renting = is_renting;
    this.is_returning = is_returning;
    this.last_reported = last_reported;
    this.eightd_has_available_keys = eightd_has_available_keys;
    this.is_charging = is_charging;
  }

  getInfo() {
    return `Name: ${this.name}\nCapacity: ${this.capacity}\nNum Bikes Available: ${this.num_bikes_available}\nNum Docks Available: ${this.num_docks_available}`;
  }
}

const translations = {
  en: {
    bixiStationStatusURL:
      "https://gbfs.velobixi.com/gbfs/en/station_status.json",
    bixiStationInformationURL:
      "https://gbfs.velobixi.com/gbfs/en/station_information.json",
    markerPopupHere: "You are here!",
    bikeAvailableString: "Bikes Available",
    dockAvailableString: "Docks Available",
    dockDisabledString: "Docks Disabled",
    capacityString: "Capacity",
  },
  fr: {
    bixiStationStatusURL:
      "https://gbfs.velobixi.com/gbfs/fr/station_status.json",
    bixiStationInformationURL:
      "https://gbfs.velobixi.com/gbfs/fr/station_information.json",
    markerPopupHere: "Vous êtes ici!",
    bikeAvailableString: "Vélos Disponibles",
    dockAvailableString: "Places Disponibles",
    dockDisabledString: "Places Désactivées",
    capacityString: "Capacité",
  },
};

// initialize variables
var currentLanguage = "en";
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map);
var bixiStationsArray = [];
var bixiIdToArrayPosDict = {};

function toggleLanguage() {
  currentLanguage = currentLanguage === "en" ? "fr" : "en";
  button = document.getElementById("language-button");
  button.textContent = currentLanguage;

  updateMarkerText();
  updateBixiStationsVisuals();
}

function updateMarkerText() {
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

  // init bixiIdToArrayPosDict
  // Note: need to treat key as object with reduce(), otherwise, js treats
  // station_id as an index instead of the key
  // bixiIdToArrayPosDict = data.data.stations.map((station, idx) => ({
  //   key: station.station_id,
  //   value: idx,
  // }));
  bixiIdToArrayPosDict = data.data.stations.reduce((acc, station, idx) => {
    acc[station.station_id] = idx;
    return acc;
  }, {});

  updateBixiAvailability();
}

function updateBixiStationsVisuals() {
  bixiStationsArray.forEach((station) => {
    stationColor = station.num_bikes_available > 0 ? "green" : "red";
    const circle = L.circle([station.lat, station.lon], {
      color: stationColor,
      fillColor: stationColor,
      fillOpacity: 0.2,
      radius: 25,
    }).addTo(map);

    // add popup with station information on hover
    const popupContent = `
    <b>${station.name}</b><br>
    ${translations[currentLanguage].capacityString} : ${station.capacity} <br>
    ${translations[currentLanguage].bikeAvailableString} : ${station.num_bikes_available} <br>
    ${translations[currentLanguage].dockAvailableString} : ${station.num_docks_available} <br>
    ${translations[currentLanguage].dockDisabledString} : ${station.num_docks_disabled} <br>
      `;
    const popup = L.popup().setContent(popupContent);
    circle.on("mouseover", function (e) {
      popup.setLatLng(e.latlng).openOn(map);
    });
    circle.on("mouseout", function () {
      map.closePopup();
    });
  });
}

async function updateBixiAvailability() {
  // fetching bixi availability JSON
  const data = await fetchJSONData(
    translations[currentLanguage].bixiStationStatusURL
  );

  // update bixi based on id pos
  data.data.stations.forEach((station) => {
    const idx = bixiIdToArrayPosDict[station.station_id];
    bixiStationsArray[idx].updateAvailability(
      station.num_bikes_available,
      station.num_ebikes_available,
      station.num_bikes_disabled,
      station.num_docks_available,
      station.num_docks_disabled,
      station.is_installed,
      station.is_renting,
      station.is_returning,
      station.last_reported,
      station.eightd_has_available_keys,
      station.is_charging
    );
  });

  updateBixiStationsVisuals();

  console.log("Updated Bixi Availability");
}

function updateMarkerPosition(e) {
  coord.updatePosition(e.latlng.lat, e.latlng.lng);

  // TODO: set popup content to see closest Bixi stations
  // marker.setLatLng(e.latlng).setPopupContent(popupContent).openOn(map);
  marker.setLatLng([coord.x, coord.y]); //.openOn(map);
  updateMarkerText();
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
