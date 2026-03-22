export class REVRoute {
  constructor(
    id_cycl,
    id_trc,
    afficheur_dynamique,
    avancement_code,
    avancement_desc,
    compteur_cyliste,
    longueur,
    nbr_voie,
    nom_arr_ville_code,
    nom_arr_ville_desc,
    protege_4s,
    rev_avancement_code,
    rev_avancement_desc,
    route_verte,
    saisons4,
    sas_velo,
    separateur_code,
    separateur_desc,
    type_voie_code,
    type_voie_desc,
    type_voie2_code,
    type_voie2_desc,
    ville_mtl,
    lineStringCoordinates,
    isVisible
  ) {
    // ID_CYCL,
    // ID_TRC,
    // AFFICHEUR_DYNAMIQUE,
    // AVANCEMENT_CODE,
    // AVANCEMENT_DESC,
    // COMPTEUR_CYLISTE,
    // LONGUEUR,
    // NBR_VOIE,
    // NOM_ARR_VILLE_CODE,
    // NOM_ARR_VILLE_DESC,
    // PROTEGE_4S,
    // REV_AVANCEMENT_CODE,
    // REV_AVANCEMENT_DESC,
    // ROUTE_VERTE,
    // SAISONS4,
    // SAS_VELO,
    // SEPARATEUR_CODE,
    // SEPARATEUR_DESC,
    // TYPE_VOIE_CODE,
    // TYPE_VOIE_DESC,
    // TYPE_VOIE2_CODE,
    // TYPE_VOIE2_DESC,
    // VILLE_MTL,

    this.id_cycl = id_cycl;
    this.id_trc = id_trc;
    this.afficheur_dynamique = afficheur_dynamique;
    this.avancement_code = avancement_code;
    this.avancement_desc = avancement_desc;
    this.compteur_cyliste = compteur_cyliste;
    this.longueur = longueur;
    this.nbr_voie = nbr_voie;
    this.nom_arr_ville_code = nom_arr_ville_code;
    this.nom_arr_ville_desc = nom_arr_ville_desc;
    this.protege_4s = protege_4s;
    this.rev_avancement_code = rev_avancement_code;
    this.rev_avancement_desc = rev_avancement_desc;
    this.route_verte = route_verte;
    this.saisons4 = saisons4;
    this.sas_velo = sas_velo;
    this.separateur_code = separateur_code;
    this.separateur_desc = separateur_desc;
    this.type_voie_code = type_voie_code;
    this.type_voie_desc = type_voie_desc;
    this.type_voie2_code = type_voie2_code;
    this.type_voie2_desc = type_voie2_desc;
    this.ville_mtl = ville_mtl;
    this.lineStringCoordinates = lineStringCoordinates;
    this.isVisible = isVisible;
  }

  addToMap(map) {
    // TODO
  }

  updateVisual() {
    // TODO
  }
}
