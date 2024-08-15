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

class ArceauxStation {
  constructor(
    _id,
    aire,
    ancrage,
    anc_num,
    base,
    categorie,
    catl_modele,
    ce_no,
    condition,
    couleur,
    date_inspection,
    element,
    empl_id,
    empl_x,
    empl_y,
    empl_z,
    intervention,
    inv_catl_no,
    inv_id,
    inv_no,
    lat,
    long,
    marq,
    materiau,
    ordre_affichage,
    parc,
    statut,
    territoire
  ) {
    this.id = _id;
    this.aire = aire;
    this.ancrage = ancrage;
    this.anc_num = anc_num;
    this.base = base;
    this.categorie = categorie;
    this.catl_modele = catl_modele;
    this.ce_no = ce_no;
    this.condition = condition;
    this.couleur = couleur;
    this.date_inspection = date_inspection;
    this.element = element;
    this.empl_id = empl_id;
    this.empl_x = empl_x;
    this.empl_y = empl_y;
    this.empl_z = empl_z;
    this.intervention = intervention;
    this.inv_catl_no = inv_catl_no;
    this.inv_id = inv_id;
    this.inv_no = inv_no;
    this.lat = lat;
    this.long = long;
    this.marq = marq;
    this.materiau = materiau;
    this.ordre_affichage = ordre_affichage;
    this.parc = parc;
    this.statut = statut;
    this.territoire = territoire;

    this.distance_from_marker_in_km = -1;
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

    this.distance_from_marker_in_km = -1;
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

MONTREAL_ARCEAUX_URL =
  "https://donnees.montreal.ca/api/3/action/datastore_search?resource_id=78dd2f91-2e68-4b8b-bb4a-44c1ab5b79b6&limit=1000";

const translations = {
  en: {
    bixiStationStatusURL:
      "https://gbfs.velobixi.com/gbfs/en/station_status.json",
    bixiStationInformationURL:
      "https://gbfs.velobixi.com/gbfs/en/station_information.json",
    markerPopupHere: "You are here!",
    lookingForBixiButtonText: "Looking for Bixi",
    notLookingForBixiButtonText: "Putting Back Bixi",
    distanceString: "Distance",
    reloadString: "reload",
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
    lookingForBixiButtonText: "Chercher un Bixi",
    notLookingForBixiButtonText: "Remettre un Bixi",
    distanceString: "Distance",
    reloadString: "rafraîchir",
    bikeAvailableString: "Vélos Disponibles",
    dockAvailableString: "Places Disponibles",
    dockDisabledString: "Places Désactivées",
    capacityString: "Capacité",
  },
};

// initialize variables
const AVAIBLE_BIKE_COLOR = "green";
const UNAVAILABLE_BIKE_COLOR = "red";
const ARCEAUX_STATIONS_COLOR = "turquoise";
const BIKE_STATION_RADIUS = 25;
const ARCEAUX_STATION_RADIUS = 15;
const NUM_DECIMAL_FORMAT = 4;
var currentLanguage = "en";
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map);
var arceauxStationsArray = [];
var arceauxIdToArrayPosDict = {};
var bixiStationsArray = [];
var bixiIdToArrayPosDict = {};
var isLookingForBixi = true;

function toggleLanguage() {
  currentLanguage = currentLanguage === "en" ? "fr" : "en";

  updateLanguageButtonText();
  updateMarkerText();
  updateBixiStationsVisuals();
  updateReloadButtonText();
}

function toggleLookingForBixiButton() {
  isLookingForBixi = !isLookingForBixi;

  updateLookingForBixiButtonText();
  updateBixiStationsVisuals();
}

function updateLookingForBixiButtonText() {
  const buttonText = isLookingForBixi
    ? translations[currentLanguage].lookingForBixiButtonText
    : translations[currentLanguage].notLookingForBixiButtonText;
  button = document.getElementById("looking-for-bixi-button");
  button.textContent = buttonText;
}

function updateLanguageButtonText() {
  button = document.getElementById("language-button");
  button.textContent = currentLanguage;
}

function updateReloadButtonText() {
  button = document.getElementById("reload-button");
  button.textContent = translations[currentLanguage].reloadString;
}

function updateMarkerText() {
  marker.setPopupContent(translations[currentLanguage].markerPopupHere);
}

function initLanguageText() {
  updateLanguageButtonText();
  updateReloadButtonText();
  updateLookingForBixiButtonText();

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
  initArceauxStationsOnMap();
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

async function initArceauxStationsOnMap() {
  const data = await fetchJSONData(MONTREAL_ARCEAUX_URL);
  arceauxStationsArray = data.result.records.map(
    (station) =>
      new ArceauxStation(
        station._id,
        station.AIRE,
        station.ANCRAGE,
        station.ANC_NUM,
        station.BASE,
        station.CATEGORIE,
        station.CATL_MODELE,
        station.CE_NO,
        station.CONDITION,
        station.COULEUR,
        station.DATE_INSPECTION,
        station.ELEMENT,
        station.EMPL_ID,
        station.EMPL_X,
        station.EMPL_Y,
        station.EMPL_Z,
        station.INTERVENTION,
        station.INV_CATL_NO,
        station.INV_ID,
        station.INV_NO,
        station.LAT,
        station.LONG,
        station.MARQ,
        station.MATERIAU,
        station.ORDRE_AFFICHAGE,
        station.PARC,
        station.STATUT,
        station.TERRITOIRE
      )
  );

  // init arceaux id to array pos dict
  arceauxIdToArrayPosDict = data.result.records.reduce((acc, station, idx) => {
    acc[station._id] = idx;
    return acc;
  }, {});

  updateArceauxStationsDistanceFromMarker();
  updateArceauxStationVisuals();
}

function updateArceauxStationVisuals() {
  arceauxStationsArray.forEach((station) => {
    // add circle
    var circle = L.circle([station.lat, station.long], {
      color: ARCEAUX_STATIONS_COLOR,
      fillColor: ARCEAUX_STATIONS_COLOR,
      fillOpacity: 0.2,
      radius: ARCEAUX_STATION_RADIUS,
    }).addTo(map);

    // add popup information on hover
    const popupContent = `
    <b>${station.parc}</b> <br>
    ${station.lat} ${station.long} <br>
    ${
      translations[currentLanguage].distanceString
    } : ${station.distance_from_marker_in_km.toFixed(
      NUM_DECIMAL_FORMAT
    )} km <br>
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
  updateBixiStationsDistanceFromMarker();
  // updateBixiStationsVisuals();
}

function updateBixiStationsVisuals() {
  bixiStationsArray.forEach((station) => {
    const getStationColor = (station, isLookingForBixi) => {
      return isLookingForBixi
        ? station.num_bikes_available > 0
          ? AVAIBLE_BIKE_COLOR
          : UNAVAILABLE_BIKE_COLOR
        : station.num_docks_available > 0
        ? AVAIBLE_BIKE_COLOR
        : UNAVAILABLE_BIKE_COLOR;
    };
    stationColor = getStationColor(station, isLookingForBixi);

    const circle = L.circle([station.lat, station.lon], {
      color: stationColor,
      fillColor: stationColor,
      fillOpacity: 0.2,
      radius: BIKE_STATION_RADIUS,
    }).addTo(map);

    // add popup with station information on hover
    const popupContent = `
    <b>${station.name}</b><br>
    ${translations[currentLanguage].capacityString} : ${station.capacity} <br>
    ${translations[currentLanguage].bikeAvailableString} : ${
      station.num_bikes_available
    } <br>
    ${translations[currentLanguage].dockAvailableString} : ${
      station.num_docks_available
    } <br>
    ${translations[currentLanguage].dockDisabledString} : ${
      station.num_docks_disabled
    } <br>
    ${
      translations[currentLanguage].distanceString
    } : ${station.distance_from_marker_in_km.toFixed(
      NUM_DECIMAL_FORMAT
    )} km <br>
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

  marker.setLatLng([coord.x, coord.y]); //.openOn(map);

  updateBixiStationsDistanceFromMarker();
  updateArceauxStationsDistanceFromMarker();

  // TODO: set popup content to see closest Bixi stations
  // marker.setLatLng(e.latlng).setPopupContent(popupContent).openOn(map);
  updateMarkerText();
}

function getClosestBixiStations() {
  // TODO

  // find available bixi stations
  availableStations = bixiStationsArray.filter(
    (station) => station.num_bikes_available > 0
  );

  // compute closest stations and display in order
}

function getDistanceBetweenCoordinatesInKM(lat1, lon1, lat2, lon2) {
  const EARTH_RADIUS = 6371;

  // convert degrees to radians
  const lat1Rad = lat1 * (Math.PI / 180);
  const lat2Rad = lat2 * (Math.PI / 180);
  const lon1Rad = lon1 * (Math.PI / 180);
  const lon2Rad = lon2 * (Math.PI / 180);

  // Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
  const dLat = lat2Rad - lat1Rad;
  const dLon = lon2Rad - lon1Rad;
  const a = Math.sin(dLat / 2) ** 2;
  const b = Math.cos(lon1) * Math.cos(lon2) * Math.sin(dLon / 2) ** 2;
  const dist = 2 * EARTH_RADIUS * Math.asin(Math.sqrt(a + b));

  return dist;
}

function updateBixiStationsDistanceFromMarker() {
  bixiStationsArray.forEach((station) => {
    station.distance_from_marker_in_km = getDistanceBetweenCoordinatesInKM(
      coord.x,
      coord.y,
      station.lat,
      station.lon
    );
  });

  updateBixiStationsVisuals();
}

function updateArceauxStationsDistanceFromMarker() {
  // FIXME
  arceauxStationsArray.forEach((station) => {
    station.distance_from_marker_in_km = getDistanceBetweenCoordinatesInKM(
      coord.x,
      coord.y,
      station.lat,
      station.long
    );
  });

  updateArceauxStationVisuals();
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
