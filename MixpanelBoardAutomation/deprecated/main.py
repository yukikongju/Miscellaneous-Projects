#!/usr/bin/env python3
"""
Adds global filters and breakdown to a Mixpanel AB test board via Playwright.

UI observations:
- Clicking "+" opens a 2-item dropdown (Filter / Breakdown).
- Clicking "Filter" opens an "Add Filter" MODAL (not an inline chip).
  The modal has a property selector, an "Is" operator, a value search+checkbox list,
  and an "Add" button to confirm.
- Clicking "Breakdown" opens a similar modal for property selection.
- Mixpanel uses deeply nested Web Components with shadow DOM throughout.
  All clicks use real mouse events at dynamically-found element coordinates.

Shadow DOM path for the "+" dropdown:
  mp-select[icon=plus] -> shadow -> mp-drop-menu -> mp-items-menu -> shadow -> li[aria-label]

Usage:
    uv run main.py

Auth:
    Run save_auth.py once first.
"""

import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright, Page

AUTH_STATE = Path("auth_state.json")
BOARD_URL = "https://mixpanel.com/project/2481461/app/boards#id=11243492"

CONFIG_PROP = "ab_home_config_slot_goalContentSuggestions"
CONFIG_VALUES = ["iOS_dayNightSuggestions_control", "iOS_dayNightSuggestions_variant"]


# ---------------------------------------------------------------------------
# Shadow DOM coordinate finders
# ---------------------------------------------------------------------------


async def find_plus_button_coords(page: Page) -> dict | None:
    """Return center {x, y} of the filter-bar + button.

    Filters by x > 250 (right of left sidebar) and y between 60-200
    to avoid the "Create New" sidebar button.
    """
    return await page.evaluate(
        """
        () => {
            function findDeep(root) {
                for (const btn of root.querySelectorAll('mp-button[icon="plus"]')) {
                    const rect = btn.getBoundingClientRect();
                    if (rect.width > 0 && rect.x > 250 && rect.y > 60 && rect.y < 200) {
                        return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
                    }
                }
                for (const el of root.querySelectorAll('*')) {
                    if (el.shadowRoot) {
                        const r = findDeep(el.shadowRoot);
                        if (r) return r;
                    }
                }
                return null;
            }
            return findDeep(document);
        }
    """
    )


async def find_dropdown_item_coords(page: Page, label: str) -> dict | None:
    """After + is clicked, return center {x, y} of 'Filter' or 'Breakdown' li."""
    return await page.evaluate(
        """
        (label) => {
            const openSel = document.querySelector('mp-select.is-open');
            if (!openSel) return null;
            const s1 = openSel.shadowRoot;
            const dm = s1?.querySelector('mp-drop-menu');
            const im = dm?.querySelector('mp-items-menu');
            const s3 = im?.shadowRoot;
            if (!s3) return null;
            // aria-label match first, text fallback
            const li = s3.querySelector(`li[aria-label="${label}"]`)
                     || Array.from(s3.querySelectorAll('li')).find(l => l.textContent?.trim() === label);
            if (!li) return null;
            const rect = li.getBoundingClientRect();
            return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
        }
    """,
        label,
    )


async def find_any_input_coords(page: Page, min_width: int = 60) -> dict | None:
    """Find the first visible <input> in the entire DOM including shadow roots."""
    return await page.evaluate(
        """
        (minW) => {
            function findDeep(root) {
                for (const inp of root.querySelectorAll('input')) {
                    const rect = inp.getBoundingClientRect();
                    if (rect.width >= minW && rect.height > 0) {
                        return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
                    }
                }
                for (const el of root.querySelectorAll('*')) {
                    if (el.shadowRoot) {
                        const r = findDeep(el.shadowRoot);
                        if (r) return r;
                    }
                }
                return null;
            }
            return findDeep(document);
        }
    """,
        min_width,
    )


async def find_text_coords(page: Page, text: str) -> dict | None:
    """Deep-search all shadow roots for a leaf element with exact text."""
    return await page.evaluate(
        """
        (text) => {
            function findDeep(root) {
                for (const el of root.querySelectorAll('*')) {
                    if (el.childElementCount === 0 && el.textContent?.trim() === text) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
                        }
                    }
                    if (el.shadowRoot) {
                        const r = findDeep(el.shadowRoot);
                        if (r) return r;
                    }
                }
                return null;
            }
            return findDeep(document);
        }
    """,
        text,
    )


async def find_button_by_text_coords(page: Page, text: str) -> dict | None:
    """Find a button/role=button element with exact text content."""
    # Try regular DOM first
    btn = page.get_by_role("button", name=text, exact=True)
    if await btn.count() > 0:
        try:
            if await btn.first.is_visible():
                box = await btn.first.bounding_box()
                if box:
                    return {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}
        except Exception:
            pass
    # Deep shadow DOM fallback
    return await page.evaluate(
        """
        (text) => {
            function findDeep(root) {
                for (const el of root.querySelectorAll('button, [role="button"]')) {
                    if (el.textContent?.trim() === text) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0) return { x: rect.x + rect.width/2, y: rect.y + rect.height/2 };
                    }
                }
                for (const el of root.querySelectorAll('*')) {
                    if (el.shadowRoot) {
                        const r = findDeep(el.shadowRoot);
                        if (r) return r;
                    }
                }
                return null;
            }
            return findDeep(document);
        }
    """,
        text,
    )


async def wait_for_open_dropdown(page: Page, timeout_ms: int = 5000) -> bool:
    for _ in range(timeout_ms // 200):
        is_open = await page.evaluate("() => document.querySelector('mp-select.is-open') !== null")
        if is_open:
            return True
        await page.wait_for_timeout(200)
    return False


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------


async def dbg(page: Page, label: str):
    try:
        path = f"screenshots/debug_{label}.png"
        await page.screenshot(path=path, timeout=8000)
        print(f"  [ss] {path}")
    except Exception as e:
        print(f"  [ss failed] {e}")


# ---------------------------------------------------------------------------
# Popup dismissal
# ---------------------------------------------------------------------------


async def dismiss_all_popups(page: Page):
    for name in ["Dismiss", "Got it", "Close", "Skip"]:
        btn = page.get_by_role("button", name=name)
        if await btn.count() > 0:
            try:
                if await btn.first.is_visible():
                    await btn.first.click()
                    await page.wait_for_timeout(600)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Core UI actions
# ---------------------------------------------------------------------------


async def click_plus_open_dropdown(page: Page):
    pos = await find_plus_button_coords(page)
    if not pos:
        raise RuntimeError("Filter bar + button not found")
    await page.mouse.click(pos["x"], pos["y"])
    print(f"  [+] ({pos['x']:.0f}, {pos['y']:.0f})")
    opened = await wait_for_open_dropdown(page)
    if not opened:
        raise RuntimeError("Dropdown did not open after clicking +")


async def click_dropdown_option(page: Page, label: str):
    pos = await find_dropdown_item_coords(page, label)
    if not pos:
        raise RuntimeError(f"'{label}' not found in dropdown")
    await page.mouse.click(pos["x"], pos["y"])
    print(f"  [{label}] ({pos['x']:.0f}, {pos['y']:.0f})")
    await page.wait_for_timeout(1000)


async def type_into_focused_input(page: Page, text: str, wait_after: int = 700):
    """Find the visible input, triple-click to select all, then type."""
    pos = None
    for _ in range(15):
        pos = await find_any_input_coords(page)
        if pos:
            break
        await page.wait_for_timeout(200)
    if pos:
        # Triple-click selects all text so the next keystroke replaces it
        await page.mouse.click(pos["x"], pos["y"], click_count=3)
        await page.wait_for_timeout(150)
    await page.keyboard.type(text, delay=40)
    await page.wait_for_timeout(wait_after)


async def click_result_item(page: Page, text: str, timeout_ms: int = 6000):
    """Wait for and click a result item with exact matching text."""
    pos = None
    for _ in range(timeout_ms // 300):
        pos = await find_text_coords(page, text)
        if pos:
            break
        await page.wait_for_timeout(300)
    if not pos:
        raise RuntimeError(f"Result item not found: '{text}'")
    await page.mouse.click(pos["x"], pos["y"])
    print(f"  [pick] '{text}' ({pos['x']:.0f}, {pos['y']:.0f})")
    await page.wait_for_timeout(500)


async def click_button(page: Page, text: str):
    """Click a button by its text label."""
    pos = None
    for _ in range(15):
        pos = await find_button_by_text_coords(page, text)
        if pos:
            break
        await page.wait_for_timeout(200)
    if not pos:
        raise RuntimeError(f"Button '{text}' not found")
    await page.mouse.click(pos["x"], pos["y"])
    print(f"  [btn] '{text}' ({pos['x']:.0f}, {pos['y']:.0f})")
    await page.wait_for_timeout(800)


# ---------------------------------------------------------------------------
# Filter / breakdown workflows
# ---------------------------------------------------------------------------


async def add_string_filter(page: Page, prop: str, values: list[str], label: str = ""):
    """Open the 'Add Filter' modal, pick a property, select values, confirm with Add."""
    print(f"\n--- Filter: {prop} = {values}")

    # Step 1: open + dropdown and click "Filter"
    await click_plus_open_dropdown(page)
    await click_dropdown_option(page, "Filter")
    await dbg(page, f"{label}_01_modal_opened")

    # Step 2: property search → select property
    await type_into_focused_input(page, prop)
    await dbg(page, f"{label}_02_prop_typed")
    await click_result_item(page, prop)
    await dbg(page, f"{label}_03_prop_selected")

    # Step 3: value search → check each value
    for i, v in enumerate(values):
        await type_into_focused_input(page, v)
        await dbg(page, f"{label}_04_{i}_val_typed")
        await click_result_item(page, v)
        await dbg(page, f"{label}_05_{i}_val_picked")

    # Step 4: confirm with "Add"
    await click_button(page, "Add")
    await dbg(page, f"{label}_06_added")


async def add_breakdown(page: Page, prop: str):
    """Open the 'Add Breakdown' flow and pick a property."""
    print(f"\n--- Breakdown: {prop}")

    await click_plus_open_dropdown(page)
    await click_dropdown_option(page, "Breakdown")

    await type_into_focused_input(page, prop)
    await page.wait_for_timeout(400)
    await click_result_item(page, prop)

    # Breakdown might also need an "Add" button
    try:
        await click_button(page, "Add")
    except RuntimeError:
        pass  # Some breakdown flows auto-close

    await page.keyboard.press("Escape")
    await page.wait_for_timeout(500)
    await dbg(page, "breakdown_done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    if not AUTH_STATE.exists():
        print("ERROR: auth_state.json not found. Run save_auth.py first.")
        sys.exit(1)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        ctx = await browser.new_context(storage_state=str(AUTH_STATE))
        page = await ctx.new_page()

        print(f"Opening: {BOARD_URL}")
        await page.goto(BOARD_URL, wait_until="load", timeout=60000)
        await page.wait_for_selector("text=AB Test - iOS - One Tap Daily Relief", timeout=15000)
        await page.wait_for_timeout(2000)
        await dismiss_all_popups(page)
        await page.wait_for_timeout(2000)
        await dismiss_all_popups(page)
        await page.wait_for_timeout(1500)
        await dbg(page, "00_board_loaded")
        print("Board loaded!")

        # Filter 1: config property with two variant values
        await add_string_filter(page, CONFIG_PROP, CONFIG_VALUES, label="config")

        # Filter 2: app_lang = 'en'
        await add_string_filter(page, "app_lang", ["en"], label="app_lang")

        # Breakdown
        await add_breakdown(page, CONFIG_PROP)

        # Save
        print("\n--- Save")
        await click_button(page, "Save")
        await page.wait_for_timeout(2000)
        await dbg(page, "final_saved")

        print(f"\nDone! {BOARD_URL}")
        input("Press Enter to close the browser...")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
