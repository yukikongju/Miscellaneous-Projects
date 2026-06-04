#!/usr/bin/env python3
"""
Automate Mixpanel AB test board creation.

Steps:
  1. Playwright  — Duplicate the template board
  2. Playwright  — Intercept the example board's API response to capture filter structure
  3. Playwright  — Add global dashboard filters + breakdown
  4. Playwright  - Change Title with nomenclature "AB Test - <platform> - <Experiment Name>"
  5. Playwright  - Add Sleep admin Link at the top of file
  6. Playwrigth - Share with

Usage:
    # First time only:
    python save_auth.py

    # Then for each experiment:
    python create_board.py
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

# ──────────────────────────────────────────────────────────────────────────────
# Project / board IDs
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ID = 2481461
TEMPLATE_DASHBOARD_ID = 8766472  # AB Test - KPI Template - Core Product Screen Home
EXAMPLE_DASHBOARD_ID = 11214293  # AB Test - AND - One Tap Daily Relief

BASE_URL = f"https://mixpanel.com/project/{PROJECT_ID}/app/boards"
AUTH_STATE_PATH = Path("auth_state.json")
SCREENSHOTS_DIR = Path("screenshots")

# ──────────────────────────────────────────────────────────────────────────────
# Experiment config  (edit per run)
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT = {
    "board_name": "AB Test - iOS - One Tap Daily Relief",
    "sleep_admin_url": "https://sleepadmin.ipnos.com/experiments/xf4UgCOfItjMT55dlM1D?tab=Informations",
    "config_property": "ab_home_config_slot_goalContentSuggestions",
    "config_values": [
        "iOS_dayNightSuggestions_control",
        "iOS_dayNightSuggestions_variant",
    ],
    "days_old_max": 14,
    "app_lang": "en",
    "platform": "ios",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


async def screenshot(page: Page, name: str) -> None:
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    await page.screenshot(path=SCREENSHOTS_DIR / f"{name}.png", full_page=False)
    print(f"  [screenshot] {name}.png")


async def goto_board(page: Page, board_id: int) -> None:
    """Navigate to a board and wait until Mixpanel's React app has rendered content."""
    await page.goto(f"{BASE_URL}#id={board_id}")
    await page.wait_for_load_state("load")
    # Wait for the board heading (h1) or any board card to appear — React renders these
    # after the initial JS bundle executes, well after the "load" event fires.
    await page.wait_for_selector(
        'h1, [class*="BoardHeader"], [class*="boardHeader"], [class*="Dashboard"]',
        timeout=30_000,
    )
    await page.wait_for_timeout(800)


async def load_auth(context: BrowserContext) -> None:
    if not AUTH_STATE_PATH.exists():
        raise FileNotFoundError("auth_state.json not found — run save_auth.py first.")
    state = json.loads(AUTH_STATE_PATH.read_text())
    await context.add_cookies(state.get("cookies", []))
    # Restore localStorage if present
    origins = state.get("origins", [])
    if origins:
        await context.add_init_script(
            f"""
            const origins = {json.dumps(origins)};
            for (const origin of origins) {{
                for (const entry of (origin.localStorage || [])) {{
                    try {{ localStorage.setItem(entry.name, entry.value); }} catch(e) {{}}
                }}
            }}
            """
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Duplicate the template board
# ──────────────────────────────────────────────────────────────────────────────


async def duplicate_board(page: Page, board_id: int, new_name: str) -> int:
    """Copy the template board. Returns the new dashboard ID."""
    await goto_board(page, board_id)
    await screenshot(page, "01_template_board")

    # Click the top-right "..." more-options button
    more_btn = page.get_by_role("button", name="More options")
    if not await more_btn.is_visible():
        more_btn = page.locator('[data-testid="board-more-options"]')
    await more_btn.click()
    await screenshot(page, "02_more_menu_open")

    # Click "Duplicate" in the dropdown
    await page.get_by_role("menuitem", name="Duplicate").click()
    await page.wait_for_url("**/boards#id=*", timeout=15_000)
    await page.wait_for_selector("h1", timeout=20_000)
    await page.wait_for_timeout(800)
    await screenshot(page, "03_duplicated_board")

    new_board_id = int(page.url.split("id=")[-1])
    print(f"  New board ID: {new_board_id}")

    await _rename_board(page, new_name)
    await _update_sleep_admin_link(page, EXPERIMENT["sleep_admin_url"])

    return new_board_id


async def _rename_board(page: Page, new_name: str) -> None:
    title = (
        page.locator('[data-testid="board-title"]').or_(page.get_by_role("heading", level=1)).first
    )
    await title.wait_for(state="visible", timeout=5_000)
    await title.click()

    if await page.locator('input[data-testid="board-title-input"]').is_visible():
        inp = page.locator('input[data-testid="board-title-input"]')
    else:
        await title.dblclick()
        inp = page.locator("input").filter(has_text="").first

    await inp.fill(new_name)
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(800)
    await screenshot(page, "04_renamed_board")
    print(f"  Renamed to: {new_name}")


async def _update_sleep_admin_link(page: Page, url: str) -> None:
    """Replace the 'SleepAdmin Link' placeholder text cell with the actual URL."""
    print(f"  Adding SleepAdmin link: {url}")

    # The template's top text cell contains "SleepAdmin Link" — click it to edit
    cell = page.get_by_text("SleepAdmin Link", exact=True).first
    await cell.wait_for(state="visible", timeout=5_000)
    await cell.click()
    await page.wait_for_timeout(400)

    # Select all existing text and replace with a hyperlink
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Meta+A")  # macOS fallback
    await page.keyboard.press("Delete")
    await page.wait_for_timeout(200)

    # Type the raw URL — Mixpanel's rich text editor auto-linkifies it
    await page.keyboard.type(url)
    await page.wait_for_timeout(300)

    # Click outside to commit the edit
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(400)
    await screenshot(page, "04b_sleep_admin_link")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Read example board filter structure via network interception
# ──────────────────────────────────────────────────────────────────────────────


async def read_example_filter_structure(page: Page) -> dict:
    """
    Navigate to the example board and intercept the API response that carries
    the board's global filter / breakdown config. Prints the raw payload so we
    can see exactly how Mixpanel encodes filters.
    """
    captured: dict = {}

    async def _on_response(response):
        url = response.url
        # Mixpanel fetches board data from endpoints like /api/app/boards/<id>
        if (
            str(EXAMPLE_DASHBOARD_ID) in url
            and "api" in url
            and response.status == 200
            and "json" in response.headers.get("content-type", "")
        ):
            try:
                data = await response.json()
                captured.update(data)
            except Exception:
                pass

    page.on("response", _on_response)

    await goto_board(page, EXAMPLE_DASHBOARD_ID)
    await screenshot(page, "05_example_board")

    page.remove_listener("response", _on_response)

    if captured:
        print("  Captured API payload keys:", list(captured.keys()))
        filter_keys = [k for k in captured if "filter" in k.lower() or "break" in k.lower()]
        if filter_keys:
            for k in filter_keys:
                print(f"  [{k}]:", json.dumps(captured[k], indent=4))
    else:
        print("  No board API response captured — filters may live in a sub-request.")
        print("  See screenshots/05_example_board.png to inspect the filter bar manually.")

    return captured


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Add global dashboard filters + breakdown
# ──────────────────────────────────────────────────────────────────────────────


async def add_board_filters(page: Page, board_id: int) -> None:
    await goto_board(page, board_id)

    cfg = EXPERIMENT

    # Filter 1: config property with two variant values
    await _add_filter(
        page,
        property_name=cfg["config_property"],
        operator="=",
        values=cfg["config_values"],
    )

    # Filter 2: days_old <= 14
    await _add_filter(
        page,
        property_name="days_old",
        operator="≤",
        values=[str(cfg["days_old_max"])],
    )

    # Filter 3: user > app_lang = 'en'
    await _add_filter(
        page,
        property_name="app_lang",
        operator="=",
        values=[cfg["app_lang"]],
        scope="user",
    )

    # Filter 4: platform = 'ios'
    await _add_filter(
        page,
        property_name="platform",
        operator="=",
        values=[cfg["platform"]],
    )

    await screenshot(page, "06_filters_added")

    # Breakdown by config property
    await _add_breakdown(page, cfg["config_property"])

    await screenshot(page, "07_breakdown_added")

    # The filter bar shows Cancel / Save — must click Save to persist
    save_btn = page.get_by_role("button", name="Save").first
    if await save_btn.is_visible():
        await save_btn.click()
        await page.wait_for_timeout(800)
        print("  Filters saved.")
    else:
        print("  Warning: Save button not found — filters may not be persisted.")

    print("  All filters and breakdown applied.")


async def _click_plus_then(page: Page, option: str) -> None:
    """
    Click the '+' button at the end of the board filter bar, then click
    either 'Filter' or 'Breakdown' in the dropdown that appears.
    """
    plus_btn = (
        page.get_by_role("button", name="+")
        .or_(page.locator('button:text-is("+")'))
        .or_(page.locator('[aria-label="Add filter or breakdown"]'))
        .or_(page.locator('[aria-label="Add"]'))
        .or_(page.locator('[data-testid="add-filter-or-breakdown"]'))
        .first
    )
    await plus_btn.wait_for(state="visible", timeout=10_000)
    await plus_btn.click()
    await page.wait_for_timeout(300)

    # Dropdown shows exactly "Filter" and "Breakdown"
    await page.get_by_text(option, exact=True).click()
    await page.wait_for_timeout(400)


async def _select_property(page: Page, property_name: str, scope: str = "event") -> None:
    """Type in the property search box and click the matching row."""
    search = (
        page.get_by_placeholder("Search properties").or_(page.get_by_placeholder("Search")).first
    )
    await search.wait_for(state="visible", timeout=5_000)
    await search.fill(property_name)
    await page.wait_for_timeout(700)

    if scope == "user":
        user_tab = (
            page.get_by_role("tab", name="User").or_(page.get_by_text("User", exact=True)).first
        )
        if await user_tab.is_visible():
            await user_tab.click()
            await page.wait_for_timeout(300)

    prop_row = (
        page.get_by_text(property_name, exact=True)
        .or_(page.locator("li", has_text=property_name))
        .first
    )
    await prop_row.wait_for(state="visible", timeout=5_000)
    await prop_row.click()
    await page.wait_for_timeout(400)


async def _add_filter(
    page: Page,
    property_name: str,
    operator: str,
    values: list[str],
    scope: str = "event",
) -> None:
    print(f"  Adding filter: {property_name} {operator} {values}")
    await _click_plus_then(page, "Filter")
    await _select_property(page, property_name, scope)

    # Change operator when not "="
    if operator not in ("=", "=="):
        op_btn = (
            page.locator('[data-testid="filter-operator"]')
            .or_(page.get_by_role("button", name="="))
            .first
        )
        if await op_btn.is_visible():
            await op_btn.click()
            await page.get_by_text(operator, exact=True).click()
            await page.wait_for_timeout(300)

    # Enter each value
    for value in values:
        val_input = (
            page.get_by_placeholder("Search values")
            .or_(page.get_by_placeholder("Enter value"))
            .or_(page.locator('input[type="text"]').last)
        )
        await val_input.wait_for(state="visible", timeout=5_000)
        await val_input.fill(value)
        await page.wait_for_timeout(500)

        val_row = page.get_by_text(value, exact=True).first
        if await val_row.is_visible():
            await val_row.click()
        else:
            await page.keyboard.press("Enter")
        await page.wait_for_timeout(200)

    await page.keyboard.press("Escape")
    await page.wait_for_timeout(300)


async def _add_breakdown(page: Page, property_name: str) -> None:
    print(f"  Adding breakdown: {property_name}")
    await _click_plus_then(page, "Breakdown")
    await _select_property(page, property_name)
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(300)
    await screenshot(page, "07b_breakdown_selected")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=80)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        await load_auth(context)
        page = await context.new_page()

        # ── Step 1: Duplicate template ──────────────────────────────────────
        print("\nStep 1: Duplicating template board...")
        new_board_id = await duplicate_board(page, TEMPLATE_DASHBOARD_ID, EXPERIMENT["board_name"])

        # ── Step 2: Inspect example board filter API structure ──────────────
        print("\nStep 2: Reading example board filter structure...")
        await read_example_filter_structure(page)

        # ── Step 3: Add global filters + breakdown ──────────────────────────
        print("\nStep 3: Adding global filters and breakdown...")
        await add_board_filters(page, new_board_id)

        # ── Final: navigate back and screenshot the completed board ───────────
        board_url = f"{BASE_URL}#id={new_board_id}"
        await goto_board(page, new_board_id)
        await screenshot(page, "08_final_board")

        print(f"\n{'─' * 60}")
        print(f"  Board name : {EXPERIMENT['board_name']}")
        print(f"  Board ID   : {new_board_id}")
        print(f"  Board URL  : {board_url}")
        print(f"  Screenshot : screenshots/08_final_board.png")
        print(f"{'─' * 60}\n")

        # Persist refreshed auth state
        await context.storage_state(path=str(AUTH_STATE_PATH))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
