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
    parts = []
    for a in auteurs:
        nom = a["nom"]
        inst = a.get("institution", "")
        parts.append(f"{nom} ({inst})" if inst else nom)
    return ", ".join(parts)


def as_blockquote(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(f"> {l}" if l.strip() else ">" for l in lines)


# ---------------------------------------------------------------------------
# Transformation de outline.md
# ---------------------------------------------------------------------------


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

        # ── Ligne de titre de présentation : **#N texte**
        title_m = re.match(r"(\*\*#(\d+)\s+.+\*\*\s*\n?)", line)
        if title_m:
            numero = title_m.group(2)
            result.append(line)
            i += 1

            # La ligne suivante (non vide) est la ligne d'auteurs à remplacer
            if i < len(lines) and lines[i].strip():
                p = presentations.get(numero, {})
                auteurs = p.get("auteurs", [])
                if auteurs:
                    result.append("\n")  # ligne vide = paragraphe séparé
                    result.append(format_authors(auteurs) + "\n")
                    filled_authors += 1
                else:
                    result.append("\n")
                    result.append(lines[i])
                i += 1
            continue

        # ── Supprimer le label "> *Description :*" (redondant dans le PDF)
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
                result.append("*Description :*\n\n")
                result.append(as_blockquote(desc) + "\n")
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
