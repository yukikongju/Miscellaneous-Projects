#!/usr/bin/env python3
"""
Run this ONCE to save your Mixpanel browser session.

A browser window will open — log in to Mixpanel manually, then press Enter
in this terminal. The session will be saved to auth_state.json and reused
by create_board.py on every subsequent run.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

AUTH_STATE_PATH = Path("auth_state.json")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://mixpanel.com/login")
        print("Log in to Mixpanel in the browser window that just opened.")
        print("Once you are fully logged in, press Enter here to save your session...")
        input()

        await context.storage_state(path=str(AUTH_STATE_PATH))
        print(f"Session saved to {AUTH_STATE_PATH}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
