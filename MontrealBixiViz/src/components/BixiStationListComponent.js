import { BixiStation } from "../models/station.js";

class BixiStationComponentListComponent extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._stations = [];
  }

  set stations(value) {
    this._stations = value;
    this.render();
  }

  render() {
    if (!this._stations) return;

    this.shadowRoot.innerHTML = `
    <style>
	.bixi-station {
	  border: 1px solid #ccc;
	  margin: 10px 0;
	  padding: 10px;
	  border-radius: 5px;
	}
    </style>
    <div>
    ${this._stations
      .map(
        (station, idx) => `
	<div class="bixi-station">
        <h2>${idx + 1} ${station.name}</h2>
        <p>ID: ${station.id}</p>
        <p>Lat: ${station.lat}</p>
        <p>Lon: ${station.lon}</p>
	<p>Distance: ${station.distance_from_marker_in_km.toFixed(4)} km</p>
      </div>
    `
      )
      .join("")}

    </div>
      `;
  }
}

customElements.define("bixi-stations-list", BixiStationComponentListComponent);
