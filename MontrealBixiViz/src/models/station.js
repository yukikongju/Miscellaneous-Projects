import { translations } from "../assets/config/translations.js";
import {
  getLanguageStoreVariable,
  getIsLookingForBixiStoreVariable,
} from "../scripts/store.js";
import L from "leaflet";

class Station {
  static NUM_DECIMAL_FORMAT = 4;
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
  }

  _getPopupContentText() {
    return ``;
  }

  _getStationColor() {
    return Station.DEFAULT_STATION_COLOR;
  }

  _initCircle(map) {
    // init shape
    this.circle = L.circle([this.lat, this.lon], {
      color: this._getStationColor(),
      fillColor: this._getStationColor(),
      fillOpacity: Station.FILL_OPACITY_SHOW,
      radius: Station.STATION_RADIUS,
    });

    // add popup with station information
    this.circle.on("mouseover", (e) => {
      this.popup.setLatLng(e.latlng).openOn(map);
    });

    this.circle.on("mouseout", () => {
      map.closePopup();
    });
  }

  addToMap(map) {
    this._initCircle(map);
    this.circle.addTo(map);
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

export class ArceauxStation extends Station {
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
      translations[getLanguageStoreVariable()].distanceString
    } : ${this.distance_from_marker_in_km.toFixed(
      Station.NUM_DECIMAL_FORMAT
    )} km <br>
      `;
  }

  _getStationColor() {
    return ArceauxStation.DEFAULT_STATION_COLOR;
  }
}

export class BixiStation extends Station {
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
    return getIsLookingForBixiStoreVariable()
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
    ${translations[getLanguageStoreVariable()].capacityString} : ${
      this.capacity
    } <br>
    ${translations[getLanguageStoreVariable()].bikeAvailableString} : ${
      this.num_bikes_available
    } <br>
    ${translations[getLanguageStoreVariable()].dockAvailableString} : ${
      this.num_docks_available
    } <br>
    ${translations[getLanguageStoreVariable()].dockDisabledString} : ${
      this.num_docks_disabled
    } <br>
    ${
      translations[getLanguageStoreVariable()].distanceString
    } : ${this.distance_from_marker_in_km.toFixed(
      Station.NUM_DECIMAL_FORMAT
    )} km <br>
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
