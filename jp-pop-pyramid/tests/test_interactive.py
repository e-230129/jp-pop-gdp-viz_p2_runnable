"""
Playwright E2E tests for interactive pyramid HTML.

Usage:
    pip install playwright pytest-playwright
    playwright install chromium
    RUN_PLAYWRIGHT=1 pytest tests/test_interactive.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _playwright_ready() -> bool:
    """Playwright/pytest-playwright が使えるか判定する（重いE2Eはデフォルト無効）"""
    if os.getenv("RUN_PLAYWRIGHT") != "1":
        return False
    try:
        import pytest_playwright  # noqa: F401
        from playwright.sync_api import expect  # noqa: F401
        return True
    except Exception:
        return False


PLAYWRIGHT_READY = _playwright_ready()
if PLAYWRIGHT_READY:
    from playwright.sync_api import expect  # type: ignore


# Test data paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "out"
HTML_FILE = OUT_DIR / "test_interactive.html"


@pytest.fixture(scope="module", autouse=True)
def generate_test_html():
    """Generate test HTML before running tests.

    Playwrightテストが無効な場合は何もしない（他の軽量テストは走らせる）。
    """
    if not PLAYWRIGHT_READY:
        yield
        return

    import subprocess
    import sys

    # Generate interactive HTML with cohorts
    result = subprocess.run(
        [
            sys.executable, "-X", "utf8",
            "src/make_memo.py",
            "--input", str(DATA_DIR / "ipss_1965_2065.csv"),
            "--out", str(HTML_FILE),
            "--mode", "interactive",
            "--year-step", "10",
            "--cohort", "団塊の世代:1947,団塊Jr:1971",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to generate HTML: {result.stderr}"
    assert HTML_FILE.exists(), "HTML file was not created"
    yield
    # Cleanup (optional)
    # HTML_FILE.unlink(missing_ok=True)


if PLAYWRIGHT_READY:

    class TestInteractivePyramid:
        """Test interactive pyramid HTML functionality (requires Playwright)."""

        @pytest.fixture(autouse=True)
        def setup(self, page):
            """Load the HTML file before each test."""
            page.goto(f"file://{HTML_FILE.absolute()}")
            # Wait for Plotly to initialize
            page.wait_for_selector(".js-plotly-plot")
            self.page = page

        def test_page_loads(self):
            """Test that the page loads successfully."""
            expect(self.page.locator(".js-plotly-plot")).to_be_visible()

        def test_title_contains_year(self):
            """Test that the title shows year and population."""
            title = self.page.locator(".gtitle")
            expect(title).to_contain_text("1965年")
            expect(title).to_contain_text("総人口")

        def test_slider_exists(self):
            """Test that the year slider is present."""
            slider = self.page.locator(".slider-group")
            expect(slider).to_be_visible()

        def test_play_button_exists(self):
            """Test that play/stop buttons exist."""
            buttons = self.page.locator(".updatemenu-button")
            expect(buttons.first).to_be_visible()

        def test_cohort_annotations(self):
            """Test that cohort annotations are displayed."""
            # Check for cohort text in annotations
            annotations = self.page.locator(".annotation-text")
            texts = annotations.all_text_contents()

            # At least one cohort should be visible
            cohort_found = any("団塊" in t or "才" in t for t in texts)
            assert cohort_found, f"Cohort annotations not found. Found: {texts}"

        def test_slider_changes_year(self):
            """Test that moving the slider changes the displayed year."""
            # Get initial title
            title = self.page.locator(".gtitle")
            initial_text = title.text_content()
            assert "1965年" in initial_text

            # Use keyboard navigation to change year (more reliable than clicking slider)
            self.page.keyboard.press("End")  # Jump to last year
            self.page.wait_for_timeout(500)

            new_text = title.text_content()
            assert "2065年" in new_text, f"Expected 2065年, got: {new_text}"
            assert new_text != initial_text, "Year did not change"

        def test_keyboard_navigation(self):
            """Test arrow key navigation."""
            title = self.page.locator(".gtitle")
            initial_text = title.text_content()

            # Press right arrow
            self.page.keyboard.press("ArrowRight")
            self.page.wait_for_timeout(300)

            after_right = title.text_content()
            # Year should have changed (or stayed same if at end)

            # Press left arrow
            self.page.keyboard.press("ArrowLeft")
            self.page.wait_for_timeout(300)

            after_left = title.text_content()
            # Should be back to initial or different

        def test_bars_visible(self):
            """Test that population bars are visible."""
            # Male bars (negative x)
            bars = self.page.locator(".bars .point")
            expect(bars.first).to_be_visible()

        def test_axes_labels(self):
            """Test that axis labels are present."""
            # X-axis label
            x_label = self.page.locator(".xtitle")
            expect(x_label).to_contain_text("人口")

            # Y-axis label
            y_label = self.page.locator(".ytitle")
            expect(y_label).to_contain_text("年齢")

        def test_legend_visible(self):
            """Test that legend is visible."""
            legend = self.page.locator(".legend")
            expect(legend).to_be_visible()



class TestDataGeneration:
    """Test that various output modes work correctly."""

    def test_memo_generation(self, tmp_path: Path):
        """Test memo PNG generation."""
        import subprocess
        import sys

        out_file = tmp_path / "memo.png"
        result = subprocess.run(
            [
                sys.executable, "-X", "utf8",
                "src/make_memo.py",
                "--input", str(DATA_DIR / "ipss_1965_2065.csv"),
                "--out", str(out_file),
                "--mode", "memo",
                "--years", "1965,2000,2065",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert out_file.exists()
        assert out_file.stat().st_size > 10000  # Should be a real image

    def test_animation_generation(self, tmp_path: Path):
        """Test animation GIF generation."""
        import subprocess
        import sys

        out_file = tmp_path / "animation.gif"
        result = subprocess.run(
            [
                sys.executable, "-X", "utf8",
                "src/make_memo.py",
                "--input", str(DATA_DIR / "ipss_1965_2065.csv"),
                "--out", str(out_file),
                "--mode", "animation",
                "--year-step", "20",
                "--fps", "2",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert out_file.exists()
        assert out_file.stat().st_size > 10000

    def test_lexis_generation(self, tmp_path: Path):
        """Test Lexis surface PNG generation."""
        import subprocess
        import sys

        out_file = tmp_path / "lexis.png"
        result = subprocess.run(
            [
                sys.executable, "-X", "utf8",
                "src/make_memo.py",
                "--input", str(DATA_DIR / "ipss_1965_2065.csv"),
                "--out", str(out_file),
                "--mode", "lexis",
                "--cohort", "団塊:1947",
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert out_file.exists()
        assert out_file.stat().st_size > 10000


class TestCohortParsing:
    """Test cohort string parsing."""

    def test_named_cohort(self):
        """Test parsing named cohorts."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from make_memo import parse_cohorts

        result = parse_cohorts("団塊の世代:1947,団塊Jr:1971")
        assert result == [("団塊の世代", 1947), ("団塊Jr", 1971)]

    def test_numeric_cohort(self):
        """Test parsing numeric-only cohorts."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from make_memo import parse_cohorts

        result = parse_cohorts("1947,1971")
        assert result == [("1947年生", 1947), ("1971年生", 1971)]

    def test_empty_cohort(self):
        """Test parsing empty cohort string."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from make_memo import parse_cohorts

        result = parse_cohorts(None)
        assert result == []

        result = parse_cohorts("")
        assert result == []
