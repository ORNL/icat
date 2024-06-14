import re

import pytest
from playwright.sync_api import Page, expect


@pytest.mark.playwright
@pytest.mark.skip
def test_thingy(page: Page):
    page.goto("https://playwright.dev/")
    expect(page).to_have_title(re.compile("Playwright"))


@pytest.mark.playwright
def test_jupyter(jupyter_server, page):
    page.goto("http://127.0.0.1:9997/lab?reset")
    expect(page).to_have_title(re.compile("JupyterLab"))

    page.get_by_text("simple_example.ipynb").dblclick()
    expect(page.locator("#Simple-Example")).to_be_visible()

    # page.locator("div#jp-main-dock-panel > div.lm-Widget.jp-Editor.jp-InputArea-editor").first.click()
    page.get_by_label("Code Cell Content", exact=True).first.click()
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")
    page.keyboard.press("Shift+Enter")

    expect(page.locator(".anchorviz")).to_be_visible(timeout=60000)

    page.get_by_role("button", name="Dictionary").click()

    expect(page.locator("div.v-data-table.softhover-table").all()[1]).to_be_visible()
    # inner_tables = page.locator(".softhover-table").all()
    # assert len(inner_tables) > 0
    # for table in inner_tables:
    #     expect(table).to_be_visible()
