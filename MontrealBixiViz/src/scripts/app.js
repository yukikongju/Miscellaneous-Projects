import { translations } from "../assets/config/translations.js";
import { Coordinate } from "../models/coordinate.js";
import { BixiStation, ArceauxStation } from "../models/station.js";
import { REVRoute } from "../models/REVRoute.js";
import { fetchJSONData } from "./httpRequest.js";
import store, {
  getLanguageStoreVariable,
  updateLanguageStoreVariable,
  getIsLookingForBixiStoreVariable,
  toggleIsLookingForBixiStoreVariable,
} from "./store.js";
import "../components/BixiStationListComponent.js";

const MONTREAL_ARCEAUX_URL =
  "https://donnees.montreal.ca/api/3/action/datastore_search?resource_id=78dd2f91-2e68-4b8b-bb4a-44c1ab5b79b6&limit=1000";
// const REV_ROUTE_URL =
//   "https://donnees.montreal.ca/api/3/action/datastore_search";
const REV_ROUTE_JSON_PATH = "assets/reseau_cyclable.geojson";

// initialize variables
const NUM_CLOSEST_BIXI_STATIONS = 5;
const coord = new Coordinate(45.5335, -73.6483); // montreal coordinates
let map = L.map("map").setView([coord.x, coord.y], 13);
let marker = L.marker([coord.x, coord.y]).addTo(map);
let arceauxStationsArray = [];
let arceauxIdToArrayPosDict = {};
let bixiStationsArray = [];
let routesREVArray = [];
let bixiIdToArrayPosDict = {};
let hasBixiStationsStatusLoaded = false;
let showBixiStations = true;
let showArceauxStations = true;
let showREVPath = true;

window.toggleLanguage = function () {
  const newLanguage = getLanguageStoreVariable() === "en" ? "fr" : "en";
  updateLanguageStoreVariable(newLanguage);
  updateTextLanguage();
};

window.toggleReloadButton = function () {
  updateBixiAvailability();
};

function updateTextLanguage() {
  updateLanguageButtonText();
  updateMarkerText();
  updateReloadButtonText();
  updateLookingForBixiButtonText();
  updateShowArceauxButtonText();
  updateShowREVPathButtonText();
  updateBixiStationsVisuals();
  updateArceauxStationVisuals();
}

window.toggleLookingForBixiButton = function () {
  toggleIsLookingForBixiStoreVariable();
  updateLookingForBixiButtonText();
  updateBixiStationsVisuals();
};

window.toggleShowREVPathButton = function () {
  showREVPath = !showREVPath;

  updateShowREVPathButtonText();
  updateREVPathVisuals();
};

window.toggleShowArceauxButton = function () {
  showArceauxStations = !showArceauxStations;

  arceauxStationsArray.forEach((station) => {
    station.isVisible = showArceauxStations;
  });

  updateShowArceauxButtonText();

  updateArceauxStationVisuals();
};

function updateREVPathVisuals() {
  // TODO
}

function updateShowREVPathButtonText() {
  const buttonText = showREVPath
    ? translations[getLanguageStoreVariable()].showREVPathText
    : translations[getLanguageStoreVariable()].hideREVPathText;
  const button = document.getElementById("show-rev-path-button");
  button.textContent = buttonText;
}

function updateLookingForBixiButtonText() {
  const buttonText = getIsLookingForBixiStoreVariable()
    ? translations[getLanguageStoreVariable()].lookingForBixiButtonText
    : translations[getLanguageStoreVariable()].notLookingForBixiButtonText;
  const button = document.getElementById("looking-for-bixi-button");
  button.textContent = buttonText;
}

function updateLanguageButtonText() {
  const button = document.getElementById("language-button");
  button.textContent = getLanguageStoreVariable();
}

function updateReloadButtonText() {
  const button = document.getElementById("reload-button");
  button.textContent = translations[getLanguageStoreVariable()].reloadString;
}

function updateMarkerText() {
  marker.setPopupContent(
    translations[getLanguageStoreVariable()].markerPopupHere
  );
}

function updateShowArceauxButtonText() {
  const button = document.getElementById("show-arceaux-button");
  button.textContent = showArceauxStations
    ? translations[getLanguageStoreVariable()].showArceauxText
    : translations[getLanguageStoreVariable()].hideArceauxText;
}

function initButtons() {
  // init Marker popup
  marker.bindPopup(translations[getLanguageStoreVariable()].markerPopupHere);

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

async function loadREVGeoJSON() {
  const data = await fetchJSONData(REV_ROUTE_JSON_PATH);

  routesREVArray = data.features.map(
    (route) =>
      new REVRoute(
        route.properties.ID_CYCL,
        route.properties.ID_TRC,
        route.properties.AFFICHEUR_DYNAMIQUE,
        route.properties.AVANCEMENT_CODE,
        route.properties.AVANCEMENT_DESC,
        route.properties.COMPTEUR_CYLISTE,
        route.properties.LONGUEUR,
        route.properties.NBR_VOIE,
        route.properties.NOM_ARR_VILLE_CODE,
        route.properties.NOM_ARR_VILLE_DESC,
        route.properties.PROTEGE_4S,
        route.properties.REV_AVANCEMENT_CODE,
        route.properties.REV_AVANCEMENT_DESC,
        route.properties.ROUTE_VERTE,
        route.properties.SAISONS4,
        route.properties.SAS_VELO,
        route.properties.SEPARATEUR_CODE,
        route.properties.SEPARATEUR_DESC,
        route.properties.TYPE_VOIE_CODE,
        route.properties.TYPE_VOIE_DESC,
        route.properties.TYPE_VOIE2_CODE,
        route.properties.TYPE_VOIE2_DESC,
        route.properties.VILLE_MTL,
        route.geometry.coordinates,
        showREVPath
      )
  );
  // console.log(routesREVArray);
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

  arceauxStationsArray.forEach((station) => {
    station.addToMap(map);
  });

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
    translations[getLanguageStoreVariable()].bixiStationInformationURL
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
        showBixiStations && hasBixiStationsStatusLoaded
      )
  );

  bixiStationsArray.forEach((station) => {
    station.addToMap(map);
  });

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
  // const isVisible = showBixiStations & hasBixiStationsStatusLoaded;
  bixiStationsArray.forEach((station) => {
    station.updateStationVisual();
  });
}

async function updateBixiAvailability() {
  // fetching bixi availability JSON
  const data = await fetchJSONData(
    translations[getLanguageStoreVariable()].bixiStationStatusURL
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

  // set popup content to see closest Bixi stations
  // updateClosestBixiStationsList();

  updateMarkerText();
}

function updateClosestBixiStationsList() {
  const closestStations = getClosestBixiStations();

  const closestStationListComponent =
    document.getElementById("closestStationList");
  closestStationListComponent.stations = closestStations;
  // console.log("Updating component: bixi-stations-list");
}

function getClosestBixiStations() {
  // find available bixi stations
  const availableStations = bixiStationsArray.filter((station) =>
    getIsLookingForBixiStoreVariable()
      ? station.num_bikes_available > 0
      : station.num_docks_available > 0
  );

  // compute closest stations and display in order
  availableStations.sort(
    (a, b) => a.distance_from_marker_in_km - b.distance_from_marker_in_km
  );

  // keep only top 5 closest
  const closestStations = availableStations.slice(0, NUM_CLOSEST_BIXI_STATIONS);

  return closestStations;
}

function updateBixiStationsDistanceFromMarker() {
  bixiStationsArray.forEach((station) => {
    station.updateDistanceFromMarkerInKm(coord.x, coord.y);
  });

  updateBixiStationsVisuals();
  updateClosestBixiStationsList();
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
  loadREVGeoJSON();
  initButtons();
  initMap();

  // add listening events
  map.on("click", onMapClick);

  // adding some components to the app
  const app = document.getElementById("app");
}

main();
