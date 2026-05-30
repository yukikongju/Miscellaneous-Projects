"""
Remplit outline.md et écrit le résultat dans outputs/outline.md :
  1. Remplace les placeholders de description restants
  2. Ajoute les institutions à côté des noms d'auteurs
"""

import os
import re

SOURCE = "cifas2026_presentations3.md"
TARGET = "outline.md"
OUTPUT = os.path.join("outputs", "outline.md")


# ---------------------------------------------------------------------------
# Parsing de la source
# ---------------------------------------------------------------------------


def parse_presentations(path: str) -> dict[str, dict]:
    """Retourne {numero -> {description, auteurs: [{nom, institution}]}}."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    presentations: dict[str, dict] = {}
    blocks = re.split(r"\n---\n", content)

    for block in blocks:
        num_m = re.search(r"\*\*Numéro de présentation:\*\*\s*(\d+)", block)
        if not num_m:
            continue
        numero = num_m.group(1)

        # Auteurs
        auteurs = []
        for line in block.splitlines():
            m = re.match(r"\s+-\s+Nom:\s*(.+?)(?:;\s*Institution:\s*(.+))?$", line)
            if m:
                auteurs.append(
                    {
                        "nom": m.group(1).strip(),
                        "institution": (m.group(2) or "").strip(),
                    }
                )

        # Description
        desc_m = re.search(r"\*\*Description:\*\*\s*\n\n(.*)", block, re.DOTALL)
        desc = desc_m.group(1).strip() if desc_m else ""
        if desc == "(non disponible)":
            desc = ""

        presentations[numero] = {"auteurs": auteurs, "description": desc}

    return presentations


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------


def format_authors(auteurs: list[dict]) -> str:
    lines = []
    for a in auteurs:
        nom = a["nom"]
        inst = a.get("institution", "")
        lines.append(f"**{nom}** — {inst}" if inst else f"**{nom}**")
    return "\\\n".join(lines)


_UNICODE_REPLACEMENTS = {
    "∙": "·",  # ∙ BULLET OPERATOR       → · middle dot
    "⋅": "·",  # ⋅ DOT OPERATOR           → · middle dot
    " ": " ",  # THIN SPACE               → espace normal
    "​": "",  # ZERO WIDTH SPACE         → supprimé
    " ": " ",  # NON-BREAKING SPACE       → espace normal
    "‑": "-",  # NON-BREAKING HYPHEN      → tiret normal
    "‒": "-",  # FIGURE DASH              → tiret
    "⁠": "",  # WORD JOINER              → supprimé
}


def normalize(text: str) -> str:
    for char, repl in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, repl)
    return text


def as_blockquote(text: str) -> str:
    lines = normalize(text).splitlines()
    return "\n".join(f"> {l}" if l.strip() else ">" for l in lines)


# ---------------------------------------------------------------------------
# Transformation de outline.md
# ---------------------------------------------------------------------------


# Jours de la semaine → ## reste ##
_DAY_RE = re.compile(r"^(LUNDI|MARDI|MERCREDI|JEUDI|VENDREDI|SAMEDI|DIMANCHE)", re.IGNORECASE)

# Catégories d'activité → ### devient ####
_CATEGORY_RE = re.compile(
    r"^(SYMPOSIUMS?|COMMUNICATIONS?\s+LIBRES?|ATELIERS?|"
    r"PRÉSENTATIONS?\s+ÉTUDIANT\w*|KIOSQUES?|CONFÉRENCES?\s+PLÉNIÈRES?)$",
    re.IGNORECASE,
)

# Créneau horaire en tête : "12 h 00 à 13 h 20 — DÎNER" → "DÎNER 12 h 00 à 13 h 20"
_TIME_FIRST_RE = re.compile(r"^(\d+\s*h\s*\d+\s*à\s*\d+\s*h\s*\d+)\s*[—–]\s*(.+)$")


def _swap_time(content: str) -> str:
    """Met le nom de l'événement avant l'heure si l'heure est en tête."""
    m = _TIME_FIRST_RE.match(content)
    if m:
        return f"{m.group(2).strip()} — {m.group(1).strip()}"
    return content


def _remap_heading(line: str) -> str:
    """
    Réécrit le niveau de heading selon la règle :
      ##  jour          → ## (inchangé)
      ##  autre (bloc)  → ###
      ### catégorie     → ####
      ### autre (créneau, événement) → ### (heure déplacée après le titre)
      ####              → #####
    """
    if line.startswith("## "):
        content = line[3:].rstrip()
        if _DAY_RE.match(content):
            return line  # ## LUNDI… → inchangé
        return f"### {_swap_time(content)}\n"  # ## BLOC… / ## heure… → ###

    if line.startswith("### "):
        content = line[4:].rstrip()
        if _CATEGORY_RE.match(content):
            return f"#### {content}\n"  # ### SYMPOSIUMS → ####
        return f"### {_swap_time(content)}\n"  # ### créneau → heure après titre

    if line.startswith("#### "):
        content = line[5:].rstrip()
        return f"##### {content}\n"  # #### salle → #####

    return line  # autres niveaux inchangés


def fill(target_path: str, output_path: str, presentations: dict[str, dict]) -> None:
    with open(target_path, encoding="utf-8") as f:
        lines = f.readlines()

    result = []
    filled_desc = 0
    filled_authors = 0
    missing = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Remapping des niveaux de heading (##, ###, ####)
        if re.match(r"^#{2,4} ", line):
            result.append(_remap_heading(line))
            i += 1
            continue

        # ── Ligne de titre : **#N Titre de la présentation** → ######
        title_m = re.match(r"\*\*#(\d+)\s+(.+?)\*\*\s*\n?", line)
        if title_m:
            numero = title_m.group(1)
            titre = title_m.group(2).strip()

            # Heading ###### avec le numéro entre parenthèses
            #  result.append(f"\n###### {titre} (#{numero})\n")
            result.append(f"\n###### # {numero} {titre}\n")
            i += 1

            # La ligne suivante (non vide) est la ligne d'auteurs originale → remplacer
            if i < len(lines) and lines[i].strip():
                p = presentations.get(numero, {})
                auteurs = p.get("auteurs", [])
                auteurs_str = format_authors(auteurs) if auteurs else lines[i].strip()
                #  result.append(f"\n\n*Auteurs*\n\n> {auteurs_str}\n")
                result.append(f"\n\n---\n\n{auteurs_str}\n\n")
                filled_authors += 1
                i += 1
            continue

        # ── Supprimer le label "> *Description :*" (on le regénère proprement)
        if re.match(r"^>\s*\*Description\s*:?\*\s*\n?$", line):
            i += 1
            continue

        # ── Placeholder de description
        ph_m = re.search(r"<!-- Insérer le résumé de la présentation #(\d+)[^-]*ici -->", line)
        if ph_m:
            numero = ph_m.group(1)
            p = presentations.get(numero, {})
            desc = p.get("description", "")
            if desc:
                #  result.append(f"\n*Description*\n\n{as_blockquote(desc)}\n")
                result.append(f"\n\nRÉSUMÉ\n\n{as_blockquote(desc)}\n\n")
                filled_desc += 1
            else:
                result.append(line)
                missing.append(numero)
            i += 1
            continue

        result.append(line)
        i += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(result)

    print(f"  {filled_authors} lignes d'auteurs mises à jour avec institutions")
    print(f"  {filled_desc} descriptions remplies")
    if missing:
        print(f"  {len(missing)} descriptions introuvables : {missing}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Lecture de {SOURCE}…")
    presentations = parse_presentations(SOURCE)
    print(f"  {len(presentations)} présentations parsées")

    print(f"Traitement de {TARGET} → {OUTPUT}…")
    fill(TARGET, OUTPUT, presentations)
    print(f"Fichier généré : {OUTPUT}")


if __name__ == "__main__":
    main()
