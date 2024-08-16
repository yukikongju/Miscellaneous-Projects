import { translations } from "../assets/config/translations.js";
import { Coordinate } from "../models/coordinate.js";
import { fetchJSONData } from "./httpRequest.js";
import store, { setLanguage } from "./store.js";

const MONTREAL_ARCEAUX_URL =
  "https://donnees.montreal.ca/api/3/action/datastore_search?resource_id=78dd2f91-2e68-4b8b-bb4a-44c1ab5b79b6&limit=1000";

// initialize variables
const NUM_DECIMAL_FORMAT = 4;
const NUM_CLOSEST_BIXI_STATIONS = 5;
// const FILL_OPACITY_SHOW = 0.2;
// let currentLanguage;
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
var map = L.map("map").setView([coord.x, coord.y], 13);
var marker = L.marker([coord.x, coord.y]).addTo(map);
var arceauxStationsArray = [];
var arceauxIdToArrayPosDict = {};
var bixiStationsArray = [];
var bixiIdToArrayPosDict = {};
var isLookingForBixi = true;
var hasBixiStationsStatusLoaded = false;
var showBixiStations = true;
var showArceauxStations = true;

// global variables management
// store.subscribe(() => {
//   currentLanguage = store.getState().language;
//   updateTextLanguage();
// });

function getCurrentLanguage() {
  return store.getState().language;
}

class Station {
  static AVAILABLE_STATION_COLOR = "green";
  static UNAVAILABLE_STATION_COLOR = "red";
  static DEFAULT_STATION_COLOR = "blue";
  static FILL_OPACITY_SHOW = 0.2;
  static STATION_RADIUS = 25;

  constructor(lat, lon, isVisible) {
    this.lat = lat;
    this.lon = lon;
    this.isVisible = isVisible;

    this.distance_from_marker_in_km = -1;

    this.circle = null;
    this.popup = L.popup().setContent(this._getPopupContentText());
    this._initCircle();
  }

  _getPopupContentText() {
    return ``;
  }

  _getStationColor() {
    return Station.DEFAULT_STATION_COLOR;
  }

  _initCircle() {
    // init shape
    this.circle = L.circle([this.lat, this.lon], {
      color: this._getStationColor(),
      fillColor: this._getStationColor(),
      fillOpacity: Station.FILL_OPACITY_SHOW,
      radius: Station.STATION_RADIUS,
    }).addTo(map);

    // add popup with station information
    this.circle.on("mouseover", (e) => {
      this.popup.setLatLng(e.latlng).openOn(map);
    });

    this.circle.on("mouseout", () => {
      map.closePopup();
    });
  }

  updateDistanceFromMarkerInKm(markerLat, markerLon) {
    const EARTH_RADIUS = 6371;

    // convert degrees to radians
    const lat1Rad = this.lat * (Math.PI / 180);
    const lat2Rad = markerLat * (Math.PI / 180);
    const lon1Rad = this.lon * (Math.PI / 180);
    const lon2Rad = markerLon * (Math.PI / 180);

    // Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
    const dLat = lat2Rad - lat1Rad;
    const dLon = lon2Rad - lon1Rad;
    const a = Math.sin(dLat / 2) ** 2;
    const b =
      Math.cos(this.lon) * Math.cos(markerLon) * Math.sin(dLon / 2) ** 2;

    this.distance_from_marker_in_km =
      2 * EARTH_RADIUS * Math.asin(Math.sqrt(a + b));
  }

  updateStationVisual() {
    // update change station color and visibility
    if (this.circle) {
      this.circle.setStyle({
        color: this._getStationColor(),
        fillColor: this._getStationColor(),
        opacity: this.isVisible ? 1 : 0,
        fillOpacity: this.isVisible ? Station.FILL_OPACITY_SHOW : 0,
        radius: Station.STATION_RADIUS,
      });
    }

    // update popup content
    this.popup.setContent(this._getPopupContentText());
  }
}

class ArceauxStation extends Station {
  static STATION_RADIUS = 15;
  static DEFAULT_STATION_COLOR = "turquoise";

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
    territoire,
    isVisible
  ) {
    super(lat, long, isVisible);
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
    // this.lat = lat;
    // this.long = long;
    this.marq = marq;
    this.materiau = materiau;
    this.ordre_affichage = ordre_affichage;
    this.parc = parc;
    this.statut = statut;
    this.territoire = territoire;
  }

  _getPopupContentText() {
    return `
    <b>${this.parc}</b> <br>
    ${this.lat} ${this.long} <br>
    ${
      translations[getCurrentLanguage()].distanceString
    } : ${this.distance_from_marker_in_km.toFixed(NUM_DECIMAL_FORMAT)} km <br>
      `;
  }

  _getStationColor() {
    return ArceauxStation.DEFAULT_STATION_COLOR;
  }
}

class BixiStation extends Station {
  static AVAIBLE_BIKE_COLOR = "green";
  static UNAVAILABLE_BIKE_COLOR = "red";
  static STATION_RADIUS = 25;

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
    station_id,
    isVisible
  ) {
    super(lat, lon, isVisible);
    this.capacity = capacity;
    this.eightd_has_key_dispenser = eightd_has_key_dispenser;
    this.electric_bike_surchage_waiver = electric_bike_surchage_waiver;
    this.external_id = external_id;
    this.has_kiosk = has_kiosk;
    this.is_charging = is_charging;
    // this.lat = lat;
    // this.lon = lon;
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

  _getStationColor() {
    // FIXME: pass isLookingForBixi as param
    return isLookingForBixi
      ? this.num_bikes_available > 0
        ? BixiStation.AVAIBLE_BIKE_COLOR
        : BixiStation.UNAVAILABLE_BIKE_COLOR
      : this.num_docks_available > 0
      ? BixiStation.AVAIBLE_BIKE_COLOR
      : BixiStation.UNAVAILABLE_BIKE_COLOR;
  }

  _getPopupContentText() {
    return `
    <b>${this.name}</b><br>
    ${translations[getCurrentLanguage()].capacityString} : ${this.capacity} <br>
    ${translations[getCurrentLanguage()].bikeAvailableString} : ${
      this.num_bikes_available
    } <br>
    ${translations[getCurrentLanguage()].dockAvailableString} : ${
      this.num_docks_available
    } <br>
    ${translations[getCurrentLanguage()].dockDisabledString} : ${
      this.num_docks_disabled
    } <br>
    ${
      translations[getCurrentLanguage()].distanceString
    } : ${this.distance_from_marker_in_km.toFixed(NUM_DECIMAL_FORMAT)} km <br>
      `;
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
}

function toggleLanguage() {
  // currentLanguage = currentLanguage === "en" ? "fr" : "en";
  const newLanguage = store.getState().language === "en" ? "fr" : "en";
  store.dispatch(setLanguage(newLanguage));

  updateTextLanguage();
}

function updateTextLanguage() {
  updateLanguageButtonText();
  updateMarkerText();
  updateReloadButtonText();
  updateLookingForBixiButtonText();
  updateShowArceauxButtonText();
  updateBixiStationsVisuals();
  updateArceauxStationVisuals();
}

function toggleLookingForBixiButton() {
  isLookingForBixi = !isLookingForBixi;

  updateLookingForBixiButtonText();
  updateBixiStationsVisuals();
}

function toggleShowArceauxButton() {
  showArceauxStations = !showArceauxStations;

  arceauxStationsArray.forEach((station) => {
    station.isVisible = showArceauxStations;
  });

  updateShowArceauxButtonText();

  updateArceauxStationVisuals();
}

function updateLookingForBixiButtonText() {
  const buttonText = isLookingForBixi
    ? translations[getCurrentLanguage()].lookingForBixiButtonText
    : translations[getCurrentLanguage()].notLookingForBixiButtonText;
  const button = document.getElementById("looking-for-bixi-button");
  button.textContent = buttonText;
}

function updateLanguageButtonText() {
  const button = document.getElementById("language-button");
  button.textContent = getCurrentLanguage();
}

function updateReloadButtonText() {
  const button = document.getElementById("reload-button");
  button.textContent = translations[getCurrentLanguage()].reloadString;
}

function updateMarkerText() {
  marker.setPopupContent(translations[getCurrentLanguage()].markerPopupHere);
}

function updateShowArceauxButtonText() {
  const button = document.getElementById("show-arceaux-button");
  button.textContent = showArceauxStations
    ? translations[getCurrentLanguage()].showArceauxText
    : translations[getCurrentLanguage()].hideArceauxText;
}

function initButtons() {
  // init Marker popup
  marker.bindPopup(translations[getCurrentLanguage()].markerPopupHere);

  // init button text
  updateTextLanguage();
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
        station.TERRITOIRE,
        showArceauxStations
      )
  );

  // init arceaux id to array pos dict
  arceauxIdToArrayPosDict = data.result.records.reduce((acc, station, idx) => {
    acc[station._id] = idx;
    return acc;
  }, {});

  updateArceauxStationsDistanceFromMarker();
}

function updateArceauxStationVisuals() {
  arceauxStationsArray.forEach((station) => {
    station.updateStationVisual();
  });
}

async function initBixiStationsOnMap() {
  const data = await fetchJSONData(
    translations[getCurrentLanguage()].bixiStationInformationURL
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
        s.station_id,
        showBixiStations & hasBixiStationsStatusLoaded
      )
  );

  // init bixiIdToArrayPosDict
  // Note: need to treat key as object with reduce() instead of map(), otherwise, js treats
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
}

function updateBixiStationsVisuals() {
  const isVisible = showBixiStations & hasBixiStationsStatusLoaded;
  bixiStationsArray.forEach((station) => {
    station.updateStationVisual();
  });
}

async function updateBixiAvailability() {
  // fetching bixi availability JSON
  const data = await fetchJSONData(
    translations[getCurrentLanguage()].bixiStationStatusURL
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

  // set visibility
  hasBixiStationsStatusLoaded = true;
  bixiStationsArray.forEach((station) => {
    station.isVisible = showBixiStations; // & hasBixiStationsStatusLoaded
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
  // find available bixi stations
  availableStations = bixiStationsArray.filter((station) =>
    isLookingForBixi
      ? station.num_bikes_available > 0
      : station.num_docks_available > 0
  );

  // compute closest stations and display in order
  availableStations.sort(
    (a, b) => a.distance_from_marker_in_km - b.distance_from_marker_in_km
  );

  // keep only top 5 closest
  closestStations = availableStations.slice(0, NUM_CLOSEST_BIXI_STATIONS); // FIXME: check if num closest station is valid

  return closestStations;
}

function updateBixiStationsDistanceFromMarker() {
  bixiStationsArray.forEach((station) => {
    station.updateDistanceFromMarkerInKm(coord.x, coord.y);
  });

  updateBixiStationsVisuals();
}

function updateArceauxStationsDistanceFromMarker() {
  arceauxStationsArray.forEach((station) => {
    station.updateDistanceFromMarkerInKm(coord.x, coord.y);
  });

  updateArceauxStationVisuals();
}

function onMapClick(e) {
  updateMarkerPosition(e);
}

function main() {
  initButtons();
  initMap();

  // add listening events
  map.on("click", onMapClick);

  // adding some components to the app
  const app = document.getElementById("app");
}

main();
