import re
import time
from playwright.sync_api import Playwright, sync_playwright, expect, Page
from typing import List


activation_point = "home"
template_url = "https://mixpanel.com/s/16LmrZ"
mixpanel_auth_file = "auth_state.json"
config_name = "ab_home_config_slot_goalContentSuggestions"
platform_name = "ios"
app_lang = "en"


def dismiss_panel(page: Page):
    page.locator("._dismiss-button_oltql_87 > .mp-button-container").click()


def duplicate_board(page: Page):
    # test 1
    page.locator(
        ".mp-control-bar-truncated-menu-container > mp-select > ._select-container_b8uke_91 > ._select-trigger_b8uke_79 > ._select-trigger-button_b8uke_104 > .mp-button-container"
    ).first.click()  # nth(3)
    page.get_by_role("listitem", name="Duplicate", exact=True).click()

    # test 2 (doesn't work)
    #  page.locator("...").click()
    #  page.get_by_role("listitem", name="Duplicate", exact=True).click()


def press_plus_button(page: Page):
    page.locator(
        "._container_t446e_248 > mp-select > ._select-container_b8uke_91 > ._select-trigger_b8uke_79 > ._select-trigger-button_b8uke_104 > .mp-button-container"
    ).click()


def debug(page: Page):
    page.pause()


def add_breakdown(page: Page, config_name: str):
    press_plus_button(page)
    page.get_by_role("listitem", name="Breakdown").click()
    page.get_by_role("textbox", name="Search...").fill(config_name)

    # wait for dropdown to populate
    page.wait_for_timeout(3000)

    page.get_by_role("textbox", name="Search...").click()
    page.get_by_role("textbox", name="Search...").press("Enter")
    debug(page)

    #  page.get_by_role("option", name=config_name, exact=False).click()


def add_app_lang_en_filter(page: Page):
    pass


def save(page: Page):
    pass


def share(page: Page, emails: List[str]):
    pass


def run(playwright: Playwright):
    #  ---- INIT ----
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(storage_state=mixpanel_auth_file)
    page = context.new_page()

    # ---- CODE ----
    page.goto(url=template_url)
    time.sleep(3)

    # dismiss
    dismiss_panel(page=page)

    # FIXME: duplicate board
    #  duplicate_board(page=page)

    # FIXME: add breakdown config => click the "+" and add the breakdown
    add_breakdown(page=page, config_name=config_name)

    # TODO: add filters

    # TODO: save

    # TODO: share

    #

    #  ---- CLOSING ----
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
