# -*- coding: utf-8 -*-
"""
test_dashboard_playwright.py

Playwrightを使ったダッシュボードの詳細インタラクションテスト
"""

import asyncio
import os
from pathlib import Path

import pytest

try:
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None


class DashboardTestResult:
    """テスト結果を格納するクラス"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    @staticmethod
    def _safe_print(text: str):
        """Print text safely, handling encoding issues."""
        try:
            print(text)
        except UnicodeEncodeError:
            # Replace problematic characters
            safe_text = text.encode('ascii', 'replace').decode('ascii')
            print(safe_text)

    def add_pass(self, test_name: str, message: str = ""):
        self.passed += 1
        self.details.append(("PASS", test_name, message))
        self._safe_print(f"  [OK] PASS: {test_name}" + (f" - {message}" if message else ""))

    def add_fail(self, test_name: str, message: str = ""):
        self.failed += 1
        self.details.append(("FAIL", test_name, message))
        self._safe_print(f"  [NG] FAIL: {test_name}" + (f" - {message}" if message else ""))

    def add_warn(self, test_name: str, message: str = ""):
        self.warnings += 1
        self.details.append(("WARN", test_name, message))
        self._safe_print(f"  [!!] WARN: {test_name}" + (f" - {message}" if message else ""))

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"結果: {self.passed}/{total} 成功, {self.warnings} 警告"


async def _basic_loading(page, result: DashboardTestResult):
    """基本的なページ読み込みテスト"""
    print("\n[1] 基本読み込みテスト")

    # Plotlyグラフの存在確認
    plotly_div = await page.query_selector(".plotly-graph-div")
    if plotly_div:
        result.add_pass("Plotlyグラフ存在")
    else:
        result.add_fail("Plotlyグラフ存在", "グラフが見つかりません")
        return False

    # スライダーの存在確認
    slider = await page.query_selector(".slider-container")
    if slider:
        result.add_pass("スライダー存在")
    else:
        result.add_warn("スライダー存在", "スライダーが見つかりません")

    # 再生ボタンの存在確認
    play_button = await page.query_selector(".updatemenu-button")
    if play_button:
        result.add_pass("再生ボタン存在")
    else:
        result.add_warn("再生ボタン存在", "再生ボタンが見つかりません")

    # タイトル確認
    title_text = await page.evaluate("""
        () => {
            const title = document.querySelector('.gtitle');
            return title ? title.textContent : '';
        }
    """)
    if title_text and "日本の人口動態" in title_text:
        result.add_pass("タイトル確認", title_text[:40] + "...")
    else:
        result.add_warn("タイトル確認", f"予期しないタイトル: {title_text[:40] if title_text else '(empty)'}")

    return True


async def _data_range(page, result: DashboardTestResult):
    """データ範囲テスト"""
    print("\n[2] データ範囲テスト")

    data_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd._fullLayout || !gd._fullLayout.sliders) return null;

            const sliders = gd._fullLayout.sliders;
            if (sliders.length === 0) return null;

            const steps = sliders[0].steps;
            const years = steps.map(s => parseInt(s.label));

            return {
                min: Math.min(...years),
                max: Math.max(...years),
                count: years.length,
                years: years
            };
        }
    """)

    if data_info:
        result.add_pass("スライダー年範囲", f"{data_info['min']}-{data_info['max']}年 ({data_info['count']}点)")

        # 2100年まであるか確認
        if data_info['max'] >= 2100:
            result.add_pass("2100年データ存在")
        else:
            result.add_fail("2100年データ存在", f"最大年: {data_info['max']}")

        # 1950年からあるか確認
        if data_info['min'] <= 1950:
            result.add_pass("1950年データ存在")
        else:
            result.add_fail("1950年データ存在", f"最小年: {data_info['min']}")

        return data_info
    else:
        result.add_fail("データ範囲取得", "スライダーデータを取得できません")
        return None


async def _slider_navigation(page, result: DashboardTestResult, data_info: dict):
    """スライダーナビゲーションテスト"""
    print("\n[3] スライダーナビゲーションテスト")

    if not data_info:
        result.add_fail("スライダーナビゲーション", "データ情報がありません")
        return

    # 特定の年に移動してテスト
    test_years = [1950, 1995, 2020, 2070, 2100]
    available_years = data_info.get('years', [])

    for target_year in test_years:
        if target_year not in available_years:
            result.add_warn(f"年移動({target_year})", "この年はデータに含まれていません")
            continue

        # 指定年に移動
        step_index = available_years.index(target_year)
        await page.evaluate(f"""
            () => {{
                const gd = document.querySelector('.plotly-graph-div');
                const sliders = gd._fullLayout.sliders;
                if (sliders.length > 0) {{
                    Plotly.animate(gd, [sliders[0].steps[{step_index}].args[0][0]], {{
                        frame: {{duration: 0, redraw: true}},
                        transition: {{duration: 0}}
                    }});
                }}
            }}
        """)
        await asyncio.sleep(0.3)

        # 現在のフレームを確認
        current_frame = await page.evaluate("""
            () => {
                const gd = document.querySelector('.plotly-graph-div');
                if (gd && gd._fullLayout && gd._fullLayout.sliders) {
                    const slider = gd._fullLayout.sliders[0];
                    return slider.active;
                }
                return -1;
            }
        """)

        if current_frame >= 0:
            result.add_pass(f"年移動({target_year})", f"ステップ{current_frame}を検知")
        else:
            result.add_warn(f"年移動({target_year})", "スライダーactive取得不可")


async def _cursor_line(page, result: DashboardTestResult):
    """カーソル線（縦線）のテスト"""
    print("\n[4] カーソル線テスト")

    # shapesの存在確認
    shapes_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd._fullLayout) return null;

            const shapes = gd._fullLayout.shapes || [];
            const cursorShapes = shapes.filter(s =>
                s.type === 'line' &&
                s.line &&
                s.line.dash === 'dash'
            );

            return {
                totalShapes: shapes.length,
                cursorShapes: cursorShapes.length,
                details: cursorShapes.map(s => ({
                    xref: s.xref,
                    yref: s.yref,
                    x0: s.x0,
                    y0: s.y0,
                    y1: s.y1
                }))
            };
        }
    """)

    if shapes_info:
        result.add_pass("shapes取得", f"総数: {shapes_info['totalShapes']}")

        # カーソル線の確認
        if shapes_info['cursorShapes'] >= 3:
            result.add_pass("カーソル線存在", f"{shapes_info['cursorShapes']}本")

            # yref="paper"の確認
            paper_refs = [d for d in shapes_info['details'] if d.get('yref') == 'paper']
            if len(paper_refs) >= 3:
                result.add_pass("yref=paper設定", "軸レンジ非依存の設定を確認")
            else:
                result.add_warn("yref=paper設定", f"{len(paper_refs)}本のみpaper参照")
        else:
            result.add_warn("カーソル線存在", f"{shapes_info['cursorShapes']}本（期待: 3本以上）")
    else:
        result.add_fail("shapes取得", "レイアウト情報を取得できません")


async def _gdp_marker(page, result: DashboardTestResult):
    """GDPマーカーの値テスト"""
    print("\n[5] GDPマーカーテスト")

    # 特定年のGDPマーカー値を確認
    test_years = [1995, 2020, 2024]

    for year in test_years:
        # まず指定年に移動
        moved = await page.evaluate(f"""
            () => {{
                const gd = document.querySelector('.plotly-graph-div');
                const sliders = gd._fullLayout.sliders;
                if (sliders.length === 0) return false;

                const steps = sliders[0].steps;
                const idx = steps.findIndex(s => parseInt(s.label) === {year});
                if (idx < 0) return false;

                Plotly.animate(gd, [steps[idx].args[0][0]], {{
                    frame: {{duration: 0, redraw: true}},
                    transition: {{duration: 0}}
                }});
                return true;
            }}
        """)

        if not moved:
            result.add_warn(f"GDPマーカー({year})", "年に移動できません")
            continue

        await asyncio.sleep(0.3)

        # GDPマーカーの値を取得
        gdp_value = await page.evaluate("""
            () => {
                const gd = document.querySelector('.plotly-graph-div');
                if (!gd || !gd.data) return null;

                // GDP現在年マーカーを探す（mode="markers+text"でname含む）
                for (const trace of gd.data) {
                    if (trace.mode === 'markers+text' && trace.y && trace.y.length === 1) {
                        // 単一点のマーカーを探す
                        const val = trace.y[0];
                        if (val > 100 && val < 1000) {  // GDP範囲
                            return val;
                        }
                    }
                }
                return null;
            }
        """)

        if gdp_value is not None:
            result.add_pass(f"GDPマーカー({year})", f"値: {gdp_value:.1f}")
        else:
            result.add_warn(f"GDPマーカー({year})", "マーカー値を取得できません")


async def _animation_play(page, result: DashboardTestResult):
    """アニメーション再生テスト"""
    print("\n[6] アニメーション再生テスト")

    # 再生ボタンを探す
    play_button = await page.query_selector(".updatemenu-button")
    if not play_button:
        result.add_warn("アニメーション再生", "再生ボタンが見つかりません")
        return

    # 初期フレームを記録
    initial_frame = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (gd && gd._fullLayout && gd._fullLayout.sliders) {
                return gd._fullLayout.sliders[0].active;
            }
            return -1;
        }
    """)

    # まず最初のフレームに戻す
    await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            const sliders = gd._fullLayout.sliders;
            if (sliders.length > 0) {
                Plotly.animate(gd, [sliders[0].steps[0].args[0][0]], {
                    frame: {duration: 0, redraw: true},
                    transition: {duration: 0}
                });
            }
        }
    """)
    await asyncio.sleep(0.2)

    # JavaScriptでアニメーション開始
    await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            Plotly.animate(gd, null, {
                frame: {duration: 100, redraw: true},
                transition: {duration: 50},
                fromcurrent: true,
                mode: 'immediate'
            });
        }
    """)

    # 少し待ってからフレームを確認
    await asyncio.sleep(0.5)

    # 現在のフレームを確認
    current_frame = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (gd && gd._fullLayout && gd._fullLayout.sliders) {
                return gd._fullLayout.sliders[0].active;
            }
            return -1;
        }
    """)

    # アニメーションを停止
    await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            Plotly.animate(gd, [], {mode: 'immediate'});
        }
    """)

    if current_frame > 0:
        result.add_pass("アニメーション再生", f"フレーム0から{current_frame}に進行")
    else:
        result.add_warn("アニメーション再生", f"フレーム変化なし (現在: {current_frame})")


async def _axis_ranges(page, result: DashboardTestResult):
    """軸範囲テスト（動的計算の確認）"""
    print("\n[7] 軸範囲テスト")

    axis_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd._fullLayout) return null;

            const layout = gd._fullLayout;
            return {
                xaxis2: layout.xaxis2 ? layout.xaxis2.range : null,
                xaxis3: layout.xaxis3 ? layout.xaxis3.range : null,
                xaxis4: layout.xaxis4 ? layout.xaxis4.range : null,
                yaxis2: layout.yaxis2 ? layout.yaxis2.range : null,
                yaxis4: layout.yaxis4 ? layout.yaxis4.range : null,
            };
        }
    """)

    if axis_info:
        # x軸範囲の確認（年パネル用: 2075以上）
        expected_max = 2075
        for axis_name in ['xaxis2', 'xaxis3']:
            if axis_info.get(axis_name):
                x_range = axis_info[axis_name]
                if x_range[1] >= expected_max:
                    result.add_pass(f"{axis_name}範囲", f"{x_range[0]:.0f}-{x_range[1]:.0f}")
                else:
                    result.add_fail(f"{axis_name}範囲", f"最大{x_range[1]:.0f}（{expected_max}未満）")
            else:
                result.add_warn(f"{axis_name}範囲", "取得できません")

        # xaxis4は相関図（Working_Ratio）- 年ではなく割合の軸
        if axis_info.get('xaxis4'):
            x_range = axis_info['xaxis4']
            # 相関図のx軸は生産年齢人口比率（50-70%程度）
            if x_range[1] < 100:  # 比率軸として妥当な範囲
                result.add_pass("xaxis4(相関図)範囲", f"{x_range[0]:.1f}-{x_range[1]:.1f}%")
            else:
                result.add_warn("xaxis4(相関図)範囲", f"予期しない範囲: {x_range}")
        else:
            result.add_warn("xaxis4範囲", "取得できません")

        # GDP y軸範囲
        if axis_info.get('yaxis2'):
            y_range = axis_info['yaxis2']
            result.add_pass("yaxis2(GDP)範囲", f"{y_range[0]:.0f}-{y_range[1]:.0f}")
    else:
        result.add_fail("軸情報取得", "レイアウト情報を取得できません")


async def _subplot_structure(page, result: DashboardTestResult):
    """サブプロット構造テスト"""
    print("\n[8] サブプロット構造テスト")

    subplot_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) return null;

            const traceTypes = {};
            gd.data.forEach((trace, i) => {
                const type = trace.type || 'scatter';
                if (!traceTypes[type]) traceTypes[type] = 0;
                traceTypes[type]++;
            });

            return {
                totalTraces: gd.data.length,
                traceTypes: traceTypes
            };
        }
    """)

    if subplot_info:
        result.add_pass("トレース総数", f"{subplot_info['totalTraces']}個")

        types = subplot_info['traceTypes']
        if types.get('bar'):
            result.add_pass("棒グラフ存在", f"{types['bar']}個")
        else:
            result.add_warn("棒グラフ存在", "見つかりません")

        scatter_count = types.get('scatter', 0) + types.get('scattergl', 0)
        if scatter_count > 0:
            result.add_pass("散布図/線グラフ存在", f"{scatter_count}個")
        else:
            result.add_warn("散布図/線グラフ存在", "見つかりません")
    else:
        result.add_fail("サブプロット情報", "取得できません")


async def _reer_overlay(page, result: DashboardTestResult):
    """REER（実質実効為替レート）オーバーレイテスト"""
    print("\n[9] REER オーバーレイテスト")

    reer_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) return null;

            const reerTraces = gd.data.filter(trace =>
                trace.name && (
                    trace.name.includes('実効為替') ||
                    trace.name.includes('REER') ||
                    trace.name.includes('為替レート')
                )
            );

            return {
                count: reerTraces.length,
                names: reerTraces.map(t => t.name),
                hasSecondaryY: reerTraces.some(t => t.yaxis === 'y3' || t.yaxis === 'y4')
            };
        }
    """)

    if reer_info and reer_info['count'] > 0:
        result.add_pass("REER トレース存在", f"{reer_info['count']}本: {', '.join(reer_info['names'][:3])}")
        if reer_info['hasSecondaryY']:
            result.add_pass("REER 副軸使用", "右軸に配置確認")
        else:
            result.add_warn("REER 副軸使用", "副軸設定を確認できません")
    else:
        result.add_warn("REER トレース存在", "REERトレースが見つかりません")


async def _working_age_breakdown(page, result: DashboardTestResult):
    """生産年齢人口内訳テスト（色とhovertemplateで確認）"""
    print("\n[10] 生産年齢人口内訳テスト")

    breakdown_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) return null;

            // 期待される色
            const expectedColors = ['#CFE2F3', '#6FA8DC', '#0B5394', '#3D85C6'];

            // barトレースで期待色を含むものを探す
            const barTraces = gd.data.filter(t => t.type === 'bar');
            let foundColors = new Set();
            let foundHovertemplates = [];

            for (const trace of barTraces) {
                if (trace.marker && trace.marker.color) {
                    const colors = Array.isArray(trace.marker.color)
                        ? trace.marker.color
                        : [trace.marker.color];
                    for (const c of colors) {
                        if (expectedColors.includes(c)) {
                            foundColors.add(c);
                        }
                    }
                }
                if (trace.hovertemplate) {
                    const ht = trace.hovertemplate;
                    if (ht.includes('20-34') || ht.includes('35-59') ||
                        ht.includes('60-64') || ht.includes('15-19')) {
                        foundHovertemplates.push(ht.split(':')[0]);
                    }
                }
            }

            return {
                barTraceCount: barTraces.length,
                foundColorCount: foundColors.size,
                foundColors: Array.from(foundColors),
                hovertemplates: foundHovertemplates
            };
        }
    """)

    if breakdown_info:
        # 色の確認
        if breakdown_info['foundColorCount'] >= 3:
            result.add_pass("生産年齢内訳配色", f"{breakdown_info['foundColorCount']}/4色検出")
        elif breakdown_info['foundColorCount'] > 0:
            result.add_warn("生産年齢内訳配色", f"{breakdown_info['foundColorCount']}/4色のみ")
        else:
            result.add_warn("生産年齢内訳配色", "期待色が見つかりません")

        # hovertemplateの確認
        if len(breakdown_info['hovertemplates']) >= 2:
            result.add_pass("生産年齢内訳ツールチップ", f"{len(breakdown_info['hovertemplates'])}種類検出")
        else:
            result.add_warn("生産年齢内訳ツールチップ", "年齢別ツールチップが見つかりません")
    else:
        result.add_warn("生産年齢内訳トレース", "バートレース情報を取得できません")


async def _total_population(page, result: DashboardTestResult):
    """総人口トレーステスト"""
    print("\n[11] 総人口トレーステスト")

    totalpop_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) return null;

            const totalPopTraces = gd.data.filter(trace =>
                trace.name && (
                    trace.name.includes('総人口') ||
                    trace.name.includes('TotalPop') ||
                    trace.name.includes('人口総数')
                )
            );

            return {
                count: totalPopTraces.length,
                names: totalPopTraces.map(t => t.name),
                hasLine: totalPopTraces.some(t => t.mode && t.mode.includes('lines')),
                lineStyle: totalPopTraces.length > 0 && totalPopTraces[0].line ?
                    totalPopTraces[0].line.dash : null
            };
        }
    """)

    if totalpop_info and totalpop_info['count'] > 0:
        result.add_pass("総人口トレース存在", f"{totalpop_info['names'][0]}")
        if totalpop_info['hasLine']:
            result.add_pass("総人口線スタイル", f"dash: {totalpop_info['lineStyle'] or 'solid'}")
    else:
        result.add_warn("総人口トレース存在", "総人口トレースが見つかりません")


async def _annotations_check(page, result: DashboardTestResult):
    """注釈（アノテーション）テスト"""
    print("\n[12] 注釈テスト")

    annotations_info = await page.evaluate("""
        () => {
            const gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd._fullLayout) return null;

            const annotations = gd._fullLayout.annotations || [];

            // 生産年齢内訳の注釈を探す
            const workingAgeAnnotations = annotations.filter(a =>
                a.text && (
                    a.text.includes('生産年齢') ||
                    a.text.includes('20-34') ||
                    a.text.includes('35-59') ||
                    a.text.includes('GDP中核')
                )
            );

            return {
                total: annotations.length,
                workingAgeCount: workingAgeAnnotations.length,
                texts: workingAgeAnnotations.map(a => a.text.substring(0, 50))
            };
        }
    """)

    if annotations_info:
        result.add_pass("注釈総数", f"{annotations_info['total']}個")
        if annotations_info['workingAgeCount'] > 0:
            result.add_pass("生産年齢注釈", f"{annotations_info['workingAgeCount']}個")
        else:
            result.add_warn("生産年齢注釈", "生産年齢関連の注釈が見つかりません")
    else:
        result.add_warn("注釈情報取得", "レイアウト情報を取得できません")


async def _js_errors(page, result: DashboardTestResult):
    """JavaScriptエラーチェック"""
    print("\n[13] JavaScriptエラーチェック")

    # コンソールエラーをキャプチャ
    errors = []

    def handle_console(msg):
        if msg.type == 'error':
            errors.append(msg.text)

    page.on('console', handle_console)

    # ページをリロードしてエラーをキャプチャ
    await page.reload()
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(0.5)

    if len(errors) == 0:
        result.add_pass("JSエラーなし", "コンソールエラーなし")
    else:
        for err in errors[:3]:
            result.add_fail("JSエラー検出", err[:100])


async def _run_dashboard():
    """ダッシュボードの詳細インタラクションテスト"""

    html_path = Path(__file__).parent.parent / "out" / "dashboard.html"

    if not html_path.exists():
        print(f"[ERROR] HTMLファイルが見つかりません: {html_path}")
        return False

    file_url = f"file:///{html_path.resolve().as_posix()}"
    print("=" * 70)
    print("ダッシュボード詳細インタラクションテスト")
    print("=" * 70)
    print(f"対象: {file_url}")

    result = DashboardTestResult()

    if async_playwright is None:
        print("[SKIP] playwright がインストールされていません")
        return True

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # ページ読み込み
        print("\n[0] ページ読み込み...")
        await page.goto(file_url)
        await page.wait_for_load_state("networkidle")
        print("  ページ読み込み完了")

        # 各テストを実行
        basic_ok = await _basic_loading(page, result)

        if basic_ok:
            data_info = await _data_range(page, result)
            await _slider_navigation(page, result, data_info)
            await _cursor_line(page, result)
            await _gdp_marker(page, result)
            await _animation_play(page, result)
            await _axis_ranges(page, result)
            await _subplot_structure(page, result)
            # 新機能テスト
            await _reer_overlay(page, result)
            await _working_age_breakdown(page, result)
            await _total_population(page, result)
            await _annotations_check(page, result)
            await _js_errors(page, result)

        # スクリーンショット保存
        print("\n[14] スクリーンショット保存...")
        screenshot_path = Path(__file__).parent.parent / "out" / "dashboard_test_final.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"  保存先: {screenshot_path}")

        await browser.close()

    # 結果サマリー
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)

    return result.failed == 0


if __name__ == "__main__":
    success = asyncio.run(_run_dashboard())
    exit(0 if success else 1)


@pytest.mark.skipif(os.getenv("RUN_PLAYWRIGHT") != "1",
                    reason="Playwrightテストは RUN_PLAYWRIGHT=1 で明示実行")
def test_dashboard_playwright():
    """pytestエントリポイント（Playwrightは明示実行）"""
    assert asyncio.run(_run_dashboard())
