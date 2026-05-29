"""
Scraper CIFAS 2026 — résumés des présentations
Source: https://event.fourwaves.com/fr/cifas2026/resumes  (12 pages)
Sortie: cifas2026_presentations2.md
"""

import asyncio
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

BASE = "https://event.fourwaves.com"
LIST_URL = "https://event.fourwaves.com/fr/cifas2026/resumes?page={page}"
TOTAL_PAGES = 12
OUTFILE = "cifas2026_presentations3.md"
CONCURRENCY = 5  # pages détail en parallèle

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


# ---------------------------------------------------------------------------
# Utilitaires texte
# ---------------------------------------------------------------------------


def clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text or "")).strip()


def clean1(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def iso_to_heure(iso: str) -> str:
    """Convertit '2026-06-17T19:50:00Z' (UTC) en '15h50' (EDT = UTC-4)."""
    m = re.match(r"\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2}):\d{2}Z", iso)
    if not m:
        return ""
    h, mn = int(m.group(1)) - 4, int(m.group(2))
    if h < 0:
        h += 24
    return f"{h}h{mn:02d}"


# ---------------------------------------------------------------------------
# Extraction depuis la page liste (article.presentation-card)
# ---------------------------------------------------------------------------


def parse_card(article) -> dict | None:
    """Extrait URL, uuid, titre, numéro depuis une card de la page liste."""
    link = article.select_one('a.presentation-card__link[href*="/resumes/"]')
    if not link:
        return None

    href = link.get("href", "")
    full_url = urljoin(BASE, href)
    m = UUID_RE.search(full_url)
    if not m:
        return None

    title_tag = article.select_one("p.presentation-card__title")
    title = clean1(title_tag.get_text()) if title_tag else ""

    num_tag = article.select_one('p.presentation-card__meta-number[data-test-id="number"]')
    numero = ""
    if num_tag:
        mn = re.search(r"\d+", num_tag.get_text())
        if mn:
            numero = mn.group(0)

    return {
        "uuid": m.group(0),
        "url": full_url,
        "titre": title,
        "numero": numero,
    }


# ---------------------------------------------------------------------------
# Extraction depuis la page détail
# ---------------------------------------------------------------------------


def parse_authors(soup: BeautifulSoup) -> list[dict]:
    """
    Extrait auteurs + institutions depuis la section presentation-header.
    Les <sup> numérotés lient chaque auteur à son affiliation.
    """
    # Construire la map sup_number -> institution
    affil_map: dict[str, str] = {}
    for li in soup.select("li.presentation-header__affiliation"):
        sup = li.find("sup", class_="presentation-header__affiliation-sup")
        sup_num = clean1(sup.get_text()) if sup else ""
        # Texte de l'institution = tout sauf le numéro sup
        raw = clean1(li.get_text(" "))
        institution = raw.replace(sup_num, "", 1).strip() if sup_num else raw
        affil_map[sup_num] = institution

    authors = []
    for li in soup.select("li.presentation-header__author"):
        name_span = li.select_one('span[data-test-id="author-name"]')
        name = clean1(name_span.get_text(" ")) if name_span else clean1(li.get_text(" "))

        sup = li.find("sup", class_="presentation-header__author-sup")
        sup_num = clean1(sup.get_text()) if sup else ""
        institution = affil_map.get(sup_num, "")

        # Seule affiliation → l'assigner même sans correspondance
        if not institution and len(affil_map) == 1:
            institution = next(iter(affil_map.values()))

        authors.append({"nom": name, "institution": institution})

    return authors


def parse_description(soup: BeautifulSoup) -> str:
    """
    Extrait la description depuis la première section sans h2 (= le résumé).
    Les sections avec h2 sont: 'Informations supplémentaires', 'Biographie',
    'Programmé dans X session', 'Discussion' — toutes ignorées.
    """
    for sec in soup.select("section.presentation-details__section"):
        if sec.find("h2", class_="presentation-details__title"):
            continue  # section titrée → pas la description
        rte = sec.select_one("div.rte")
        if rte:
            return clean(rte.get_text("\n"))
    return ""


def parse_schedule(soup: BeautifulSoup) -> dict:
    """Extrait journée, heure (format 15h50 - 16h50), bloc depuis la section horaire."""
    journee = ""
    heure = ""
    bloc = ""

    date_tag = soup.select_one("h3.presentation-details__schedule-date")
    if date_tag:
        journee = clean1(date_tag.get_text())

    times = soup.select("p.presentation-details__schedule-slot-time time[datetime]")
    if len(times) >= 2:
        start = iso_to_heure(times[0].get("datetime", ""))
        end = iso_to_heure(times[1].get("datetime", ""))
        if start and end:
            heure = f"{start} - {end}"

    bloc_tag = soup.select_one(
        'p.presentation-details__schedule-slot-title[data-test-id="schedule-slot-title"]'
    )
    if bloc_tag:
        text = clean1(bloc_tag.get_text())
        bm = re.search(r"Bloc\s+(\d+\.[A-Z])", text)
        if bm:
            bloc = bm.group(1)

    return {"journee": journee, "heure": heure, "bloc": bloc}


def enrich_from_detail(soup: BeautifulSoup, item: dict) -> dict:
    """Complète un item avec les données de la page détail."""
    h1 = soup.select_one("h1.presentation-header__title")
    if h1:
        t = clean1(h1.get_text(" "))
        if t:
            item["titre"] = t

    item["auteurs"] = parse_authors(soup)
    item["description"] = parse_description(soup)

    sched = parse_schedule(soup)
    for k in ("journee", "heure", "bloc"):
        if not item.get(k):
            item[k] = sched[k]
        elif not item[k] and sched[k]:
            item[k] = sched[k]

    return item


# ---------------------------------------------------------------------------
# Navigation Playwright
# ---------------------------------------------------------------------------


async def load_soup(page, url: str) -> BeautifulSoup:
    await page.goto(url, wait_until="networkidle", timeout=60_000)
    await page.wait_for_timeout(800)
    return BeautifulSoup(await page.content(), "lxml")


# ---------------------------------------------------------------------------
# Phase 1 : collecter les cards depuis toutes les pages liste
# ---------------------------------------------------------------------------


async def collect_listing(page) -> list[dict]:
    items: dict[str, dict] = {}

    for page_num in range(1, TOTAL_PAGES + 1):
        url = LIST_URL.format(page=page_num)
        soup = await load_soup(page, url)

        for article in soup.select("article.presentation-card"):
            card = parse_card(article)
            if card and card["uuid"] not in items:
                items[card["uuid"]] = card

        print(f"  Page {page_num:2d}/{TOTAL_PAGES} — {len(items)} présentations collectées")

    return list(items.values())


# ---------------------------------------------------------------------------
# Phase 2 : enrichir depuis les pages détail (en parallèle)
# ---------------------------------------------------------------------------


async def enrich_worker(semaphore, browser, item: dict, idx: int, total: int) -> dict:
    async with semaphore:
        ctx = await browser.new_context(locale="fr-CA")
        pg = await ctx.new_page()
        try:
            soup = await load_soup(pg, item["url"])
            enrich_from_detail(soup, item)
            print(f"  [{idx:3d}/{total}] OK  {item.get('titre', '')[:60]}")
        except Exception as exc:
            print(f"  [{idx:3d}/{total}] ERR {item['url']} → {exc}")
        finally:
            await ctx.close()
    return item


async def enrich_all(browser, items: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(items)
    tasks = [
        enrich_worker(semaphore, browser, item, idx, total) for idx, item in enumerate(items, 1)
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Génération du fichier Markdown
# ---------------------------------------------------------------------------


def write_markdown(items: list[dict]) -> None:
    def sort_key(x):
        return (
            x.get("journee", ""),
            x.get("heure", ""),
            x.get("bloc", ""),
            int(x["numero"]) if str(x.get("numero", "")).isdigit() else 9999,
            x.get("titre", ""),
        )

    items.sort(key=sort_key)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("# Présentations CIFAS 2026\n\n")

        for i, item in enumerate(items, 1):
            titre = (item.get("titre") or "").strip()
            f.write(f"## {i}. {titre}\n\n")
            f.write(f"- **Titre:** {titre}\n")
            f.write(f"- **Journée:** {item.get('journee', '')}\n")
            f.write(f"- **Heure:** {item.get('heure', '')}\n")
            f.write(f"- **Bloc:** {item.get('bloc', '')}\n")
            f.write(f"- **Numéro de présentation:** {item.get('numero', '')}\n")
            f.write(f"- **Lien:** {item.get('url', '')}\n")

            f.write("- **Auteurs:**\n")
            auteurs = item.get("auteurs") or []
            if auteurs:
                for a in auteurs:
                    inst = f"; Institution: {a['institution']}" if a.get("institution") else ""
                    f.write(f"  - Nom: {a.get('nom', '')}{inst}\n")
            else:
                f.write("  - (non disponible)\n")

            desc = (item.get("description") or "").strip()
            f.write("\n**Description:**\n\n")
            f.write(desc if desc else "(non disponible)")
            f.write("\n\n---\n\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        list_page = await browser.new_page(locale="fr-CA")
        print("=== Phase 1 : collecte des liens depuis les pages liste ===")
        items = await collect_listing(list_page)
        await list_page.close()
        print(f"→ {len(items)} présentations trouvées\n")

        print(f"=== Phase 2 : extraction des détails ({CONCURRENCY} en parallèle) ===")
        items = await enrich_all(browser, items)

        await browser.close()

    print(f"\n=== Écriture : {OUTFILE} ===")
    write_markdown(items)
    print(f"Terminé — {len(items)} présentations exportées dans {OUTFILE}")


if __name__ == "__main__":
    asyncio.run(main())
