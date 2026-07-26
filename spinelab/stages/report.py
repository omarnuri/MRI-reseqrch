"""Stage `report`: one HTML report, assembled only from the stage JSON files.

Design rules, each of them a reaction to something the old report did:

* every section carries an evidence badge, and a HEURISTIC section may not
  contain a conclusion — only a measurement plus what would be needed to
  interpret it;
* nothing is hard-coded: if a stage did not run, its section says so and why;
* the limitations of the *study* (missing sequences, wrong plane for the
  question) are stated at the top, not buried at the bottom;
* no filename, patient name or accession number is rendered into the report.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from .. import DISCLAIMER_RU
from ..evidence import Evidence, Status, badge_html
from ..pipeline import Context, StageResult
from ..utils import write_json

CSS = """
body{font-family:'Segoe UI',Tahoma,sans-serif;line-height:1.55;max-width:940px;margin:0 auto;
padding:24px;color:#1f2933;background:#fff}
h1{border-bottom:2px solid #1f2933;padding-bottom:10px}
h2{margin-top:34px;border-bottom:1px solid #e4e7eb;padding-bottom:6px}
.disclaimer{background:#fff4f4;border-left:4px solid #ef4444;padding:12px 16px;margin:16px 0}
.box{background:#f7f9fb;border-left:4px solid #2563eb;padding:10px 15px;margin:12px 0}
.warn{background:#fffaf0;border-left:4px solid #f59e0b;padding:10px 15px;margin:12px 0}
.mute{color:#647084;font-size:13px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border-bottom:1px solid #e4e7eb;padding:7px 10px;text-align:left}
th{background:#f2f5f8}
code{background:#f2f5f8;padding:1px 5px;border-radius:4px;font-size:13px}
img{max-width:100%;border:1px solid #e4e7eb;border-radius:6px}
ul{margin:8px 0 8px 18px}
"""


#: Section numbers are assigned as sections are emitted, never written by hand:
#: hand-numbering already produced two sections called "9".
_SECTION_COUNTER = {"n": 0}


def _h2(title: str) -> str:
    _SECTION_COUNTER["n"] += 1
    return f"<h2>{_SECTION_COUNTER['n']}. {html.escape(title)}</h2>"


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    _SECTION_COUNTER["n"] = 0
    sections: list[str] = []

    ingest = ctx.stage_data("ingest")
    picks = ingest.get("picks", {})
    sections.append(_study_section(ingest, picks))
    sections.append(_fatsat_qc_section(ctx))
    figure = _try_figure(ctx)
    if figure:
        sections.append(figure)
    sections.append(_geometry_section(ctx))
    sections.append(_facets_axial_section(ctx))
    sections.append(_posterior_section(ctx))
    sections.append(_marrow_section(ctx))
    sections.append(_muscle_section(ctx))
    sections.append(_canal_section(ctx))
    sections.append(_disc_section(ctx))
    sections.append(_agreement_section(ctx))
    sections.append(_reliability_section(ctx))
    sections.append(_not_assessable_section(ctx, picks))
    sections.append(_status_section(ctx))

    body = "\n".join(s for s in sections if s)
    doc = (
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Отчёт spinelab — {html.escape(cfg.subject_id)}</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Скрининговый отчёт по МРТ позвоночника</h1>"
        f"<div class='disclaimer'><b>Важно.</b> {html.escape(DISCLAIMER_RU)}</div>"
        "<div class='box'><b>Как читать значки:</b><br>"
        f"{badge_html(Evidence.MODEL)} — результат обученной сегментационной модели.<br>"
        f"{badge_html(Evidence.MEASUREMENT)} — геометрическое измерение по такой маске: "
        "числу можно верить, клинический смысл определяет врач.<br>"
        f"{badge_html(Evidence.HEURISTIC)} — пороги по яркости. Чувствительны к типу "
        "последовательности и неоднородности поля. Только для сортировки, что смотреть первым.<br>"
        f"{badge_html(Evidence.NOT_DIAGNOSTIC)} — измерено, но на этих данных не "
        "интерпретируется. Не используйте для выводов.</div>"
        f"{body}"
        f"<p class='mute'>spinelab {ctx.results.get('report', StageResult('report', Status.OK)).spinelab_version}"
        f" · subject <code>{html.escape(cfg.subject_id)}</code></p>"
        "</body></html>"
    )

    out = cfg.results_dir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    write_json(cfg.results_dir / "findings.json", _findings_bundle(ctx))
    return StageResult(name="report", status=Status.OK, evidence=Evidence.MEASUREMENT,
                       data={"report_html": str(out),
                             "findings_json": str(cfg.results_dir / "findings.json")},
                       artifacts=[str(out)])


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _study_section(ingest: dict, picks: dict) -> str:
    rows = []
    for s in ingest.get("series", []):
        if s.get("localizer"):
            continue
        rows.append(
            f"<tr><td>{html.escape(s.get('description') or s.get('name',''))}</td>"
            f"<td>{html.escape(s.get('sequence_label',''))}</td>"
            f"<td>{html.escape(s.get('plane',''))}</td>"
            f"<td>{s.get('n_slices','')}</td>"
            f"<td>{'×'.join(str(v) for v in (s.get('voxel_mm') or []))} мм</td></tr>"
        )
    table = ("<table><tr><th>Серия</th><th>Контраст</th><th>Плоскость</th>"
             f"<th>Срезов</th><th>Воксель</th></tr>{''.join(rows)}</table>"
             if rows else "<p class='mute'>Инвентаризация серий недоступна.</p>")
    limits = picks.get("limitations") or []
    limit_html = ("<div class='warn'><b>Ограничения этого исследования:</b><ul>"
                  + "".join(f"<li>{html.escape(x)}</li>" for x in limits)
                  + "</ul></div>") if limits else ""
    return f"{_h2('Что было в исследовании')}{table}{limit_html}"


def _geometry_section(ctx: Context) -> str:
    data, note = _stage(ctx, "geometry")
    if note:
        return f"{_h2('Геометрия позвонков')} {badge_html(Evidence.MEASUREMENT)}{note}"
    rows = "".join(
        f"<tr><td>{html.escape(v['name'])}</td>"
        f"<td>{v.get('anterior_height_mm','—')}</td>"
        f"<td>{v.get('posterior_height_mm','—')}</td>"
        f"<td>{v.get('wedge_angle_deg','—')}</td></tr>"
        for v in data.get("per_vertebra", [])
    )
    verdict = (
        f"Есть {data['longest_run_at_or_above_threshold']} смежных позвонка(ов) с "
        f"клиновидностью ≥ {data['wedge_threshold_deg']}° — это соответствует правилу "
        f"«{html.escape(data['scheuermann_rule'])}». Требуется оценка врача."
        if data.get("scheuermann_pattern") else
        f"Паттерна ≥ {data.get('wedge_threshold_deg')}° на "
        f"{data.get('scheuermann_rule','').split()[-2] if data.get('scheuermann_rule') else '3+'} "
        f"смежных позвонках нет (максимум {data.get('max_wedge_angle_deg')}°)."
    )
    curvature = data.get("curvature") or {}
    curv_html = ""
    if curvature:
        curv_html = (
            f"<p>Угол цепочки центроидов грудного отдела: "
            f"<b>{curvature.get('chain_angle_deg','—')}°</b>. "
            f"Максимальное отклонение от прямой: "
            f"<b>{round(curvature.get('max_lateral_deviation_mm', 0), 1)} мм</b>.<br>"
            f"<span class='mute'>{html.escape(curvature.get('chain_angle_note',''))}</span></p>"
        )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in data.get("notes", []))
    return (
        f"{_h2('Геометрия позвонков')} {badge_html(Evidence.MEASUREMENT)}"
        f"<table><tr><th>Уровень</th><th>Передняя высота, мм</th>"
        f"<th>Задняя высота, мм</th><th>Клиновидность, °</th></tr>{rows}</table>"
        f"{curv_html}<div class='box'><b>Что показывают числа:</b> {verdict}</div>"
        + (f"<ul class='mute'>{notes}</ul>" if notes else "")
    )


def _posterior_section(ctx: Context) -> str:
    data, note = _stage(ctx, "posterior")
    level = Evidence.HEURISTIC if (data or {}).get("fat_suppressed") else Evidence.NOT_DIAGNOSTIC
    head = f"{_h2('Фасеточные и рёберно-позвоночные зоны')} {badge_html(level)}"
    if note:
        return head + note
    blocks = []
    for group, entry in (data.get("groups") or {}).items():
        title = {"facet": "Фасеточные (суставные) отростки",
                 "costal_process": "Поперечные / рёберные отростки"}.get(group, group)
        if entry.get("status") != "measured":
            blocks.append(f"<div class='warn'><b>{title}:</b> сравнение сторон не выполнено — "
                          f"{html.escape(str(entry.get('reason') or entry.get('status')))}</div>")
            continue
        sides = entry["sides"]
        cmp_bright = entry["bright_fraction_comparison"]
        cmp_median = entry["median_intensity_comparison"]
        threshold_note = (f"<p class='mute'>Порог яркости: {entry.get('threshold_value')} "
                          f"(источник: {html.escape(str(entry.get('threshold_reference')))}).</p>")
        rows = "".join(
            f"<tr><td>{'слева' if k == 'left' else 'справа'}</td>"
            f"<td>{v['region_voxels']}</td><td>{v['in_fov_coverage']}</td>"
            f"<td>{v['median_intensity']}</td><td>{v['bright_voxels']}</td>"
            f"<td>{v['bright_fraction']}</td></tr>"
            for k, v in sides.items()
        )
        per_level = entry.get("per_level") or []
        pl_rows = "".join(
            f"<tr><td>{html.escape(r['level'])}</td>"
            f"<td>{r['left_bright_voxels']}/{r['left_region_voxels']}</td>"
            f"<td>{r['right_bright_voxels']}/{r['right_region_voxels']}</td>"
            f"<td>{(r['comparison'] or {}).get('ratio','—')}×</td>"
            f"<td>{(r['comparison'] or {}).get('higher_side','—')}</td></tr>"
            for r in per_level
        )
        pl_html = (f"<p class='mute'>По уровням:</p><table><tr><th>Уровень</th>"
                   f"<th>Слева ярких/всего</th><th>Справа ярких/всего</th>"
                   f"<th>Отношение</th><th>Больше</th></tr>{pl_rows}</table>") if pl_rows else ""
        blocks.append(
            f"<h3>{title}</h3>{threshold_note}"
            f"<table><tr><th>Сторона</th><th>Вокселей в зоне</th><th>Покрытие FOV</th>"
            f"<th>Медиана сигнала</th><th>Ярких вокселей</th><th>Доля ярких</th></tr>{rows}</table>"
            f"<p>Доля ярких вокселей: отношение <b>{cmp_bright.get('ratio')}×</b>, "
            f"разница {cmp_bright.get('diff_pct')}%, выше — {cmp_bright.get('higher_side')}. "
            f"Медиана сигнала: отношение {cmp_median.get('ratio')}× "
            f"(разница {cmp_median.get('diff_pct')}%).</p>{pl_html}"
        )
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return head + "".join(blocks) + f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>"


def _marrow_section(ctx: Context) -> str:
    data, note = _stage(ctx, "marrow")
    head = f"{_h2('Сигнал костного мозга тел позвонков')} {badge_html(Evidence.HEURISTIC)}"
    if note:
        return head + note
    rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['outlier_voxels']}</td>"
        f"<td>{r['outlier_fraction']}</td><td>{r.get('relative_median_vs_cohort','—')}</td>"
        f"<td>{r['in_fov_coverage']}</td></tr>"
        for r in data.get("regions", [])
    )
    excluded = data.get("excluded") or []
    exc = ("<p class='mute'>Исключены из сравнения: "
           + ", ".join(f"{html.escape(e['name'])} ({html.escape(e['excluded_reason'])})"
                       for e in excluded) + "</p>") if excluded else ""
    top = data.get("top_candidates") or []
    top_html = (f"<div class='box'><b>Куда смотреть первым:</b> {', '.join(html.escape(t) for t in top)}. "
                "Это порядок просмотра, а не находка.</div>"
                if top else "<div class='box'>Ни один уровень не выделяется по этому критерию.</div>")
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return (head + f"<p class='mute'>Последовательность: {html.escape(str(data.get('sequence_label')))} "
            f"({html.escape(str(data.get('plane')))}).</p>"
            f"<table><tr><th>Уровень</th><th>Ярких вокселей</th><th>Доля</th>"
            f"<th>Медиана / медиана по всем</th><th>Покрытие FOV</th></tr>{rows}</table>"
            f"{exc}{top_html}<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _muscle_section(ctx: Context) -> str:
    data, note = _stage(ctx, "muscles")
    head = f"{_h2('Паравертебральные мышцы')} {badge_html(Evidence.MEASUREMENT)}"
    if note:
        return head + note
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{m['volume_left_cm3']}</td>"
        f"<td>{m['volume_right_cm3']}</td>"
        f"<td>{(m['comparison'] or {}).get('diff_pct','—')}%</td>"
        f"<td>{(m['comparison'] or {}).get('higher_side','—')}</td></tr>"
        for name, m in (data.get("muscles") or {}).items()
    )
    fat = data.get("fat_infiltration") or {}
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return (head + f"<table><tr><th>Мышца</th><th>Объём слева, см³</th><th>Объём справа, см³</th>"
            f"<th>Разница</th><th>Больше</th></tr>{rows}</table>"
            f"<p>Суммарный объём в пределах FOV: <b>{data.get('total_paraspinal_volume_cm3')} см³</b>.</p>"
            f"<div class='warn'><b>Жировая инфильтрация не измерялась.</b> "
            f"{html.escape(str(fat.get('reason','')))}</div>"
            f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _canal_section(ctx: Context) -> str:
    data, note = _stage(ctx, "canal")
    head = f"{_h2('Позвоночный канал')} {badge_html(Evidence.MEASUREMENT)}"
    if note:
        return head + note
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    verdict = ("Сужение относительно собственной медианы превышает порог "
               f"{data.get('narrowing_threshold_pct')}% — нужна проверка врачом."
               if data.get("exceeds_threshold") else
               "Выраженного относительного сужения по этому критерию нет.")
    return (head + f"<ul><li>Медиана площади: <b>{data.get('median_area_mm2')} мм²</b></li>"
            f"<li>10-й перцентиль: {data.get('p10_area_mm2')} мм²</li>"
            f"<li>Минимум: {data.get('min_area_mm2')} мм² "
            f"(срез {data.get('slice_index_of_min')})</li>"
            f"<li>Максимальное сужение: <b>{data.get('max_narrowing_pct')}%</b></li>"
            f"<li>Измерено срезов: {data.get('n_slices_measured')}</li></ul>"
            f"<div class='box'>{verdict}</div>"
            f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _disc_section(ctx: Context) -> str:
    data, note = _stage(ctx, "discs")
    head = f"{_h2('Диски: относительный сигнал')} {badge_html(Evidence.MEASUREMENT)}"
    if note:
        return head + note
    rows = "".join(
        f"<tr><td>{html.escape(d['level'])}</td><td>{d['median_signal']}</td>"
        f"<td>{d.get('relative_to_cohort','—')}</td><td>{d['darkness_rank']}</td></tr>"
        for d in data.get("discs", [])
    )
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return (head + f"<table><tr><th>Уровень</th><th>Медиана T2</th>"
            f"<th>Отн. к медиане по всем</th><th>Ранг (1 = самый тёмный)</th></tr>{rows}</table>"
            f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _agreement_section(ctx: Context) -> str:
    data, note = _stage(ctx, "agreement")
    head = f"{_h2('Согласие двух инструментов')} {badge_html(Evidence.MEASUREMENT)}"
    if note:
        return head + note
    ok = data.get("agreement_ok")
    return (head + f"<p>Dice по спинному мозгу (SPINEPS против TotalSpineSeg): "
            f"<b>{data.get('dice')}</b> при пороге {data.get('threshold')}.</p>"
            + ("<div class='box'>Инструменты согласуются — маски можно использовать.</div>" if ok else
               "<div class='warn'>Согласие низкое: числа, зависящие от этих масок, "
               "требуют визуальной проверки.</div>"))


def _fatsat_qc_section(ctx: Context) -> str:
    """Whether the fat suppression worked — this gates everything about oedema."""
    data, note = _stage(ctx, "fatsat_qc")
    head = f"{_h2('Работает ли подавление жира')} {badge_html(Evidence.HEURISTIC)}"
    if note:
        return head + note
    fat = data.get("fatsat_stats") or {}
    ctl = data.get("control_stats") or {}
    ok = data.get("suppression_effective")
    rows = (
        f"<tr><td>{html.escape(str(data.get('fatsat_label')))} "
        f"({html.escape(str(data.get('fatsat_plane')))})</td>"
        f"<td>{fat.get('rim_p90','—')}</td><td>{fat.get('core_median','—')}</td>"
        f"<td><b>{fat.get('rim_to_core_ratio','—')}</b></td></tr>"
    )
    if ctl:
        rows += (f"<tr><td>обычная T2 (контроль)</td><td>{ctl.get('rim_p90','—')}</td>"
                 f"<td>{ctl.get('core_median','—')}</td>"
                 f"<td>{ctl.get('rim_to_core_ratio','—')}</td></tr>")
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return (head
            + "<p class='mute'>Подкожный жир образует полосу под кожей. Если жир подавлен, "
              "эта полоса не ярче глубоких тканей. Сравнение того же измерения на двух "
              "сериях одного исследования снимает вопрос о масштабе сигнала.</p>"
            + f"<table><tr><th>Серия</th><th>p90 полосы</th><th>медиана ядра</th>"
              f"<th>отношение</th></tr>{rows}</table>"
            + (f"<div class='{'box' if ok else 'warn'}'><b>Вывод:</b> "
               f"{html.escape(str(data.get('verdict')))}</div>")
            + f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _facets_axial_section(ctx: Context) -> str:
    """Per-joint facet measurement on the axial series."""
    data, note = _stage(ctx, "facets_axial")
    head = f"{_h2('Фасеточные суставы по уровням (аксиальная T2)')} {badge_html(Evidence.HEURISTIC)}"
    if note:
        return head + note
    rows = []
    for joint in data.get("joints", []):
        left = (joint.get("sides") or {}).get("left") or {}
        right = (joint.get("sides") or {}).get("right") or {}
        cmp_bright = (joint.get("comparison") or {}).get("bright_fraction") or {}
        comparable = joint.get("comparable")
        rows.append(
            f"<tr><td>{html.escape(joint['joint'])}</td>"
            f"<td>{left.get('interface_volume_mm3','—')}</td>"
            f"<td>{right.get('interface_volume_mm3','—')}</td>"
            f"<td>{left.get('bright_fraction','—')}</td>"
            f"<td>{right.get('bright_fraction','—')}</td>"
            f"<td>{cmp_bright.get('ratio','—') if comparable else '—'}</td>"
            f"<td>{(cmp_bright.get('higher_side') or '—') if comparable else 'не сравнимо'}</td></tr>"
        )
    top = data.get("largest_side_difference") or []
    top_html = ("<div class='box'><b>Наибольшая разница сторон:</b> "
                + "; ".join(f"{html.escape(t['joint'])} — больше {t['higher_side']} "
                            f"(+{t['diff_pct']}%)" for t in top)
                + ". Это порядок просмотра для врача, не находка.</div>") if top else ""
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in data.get("interpretation_limits", []))
    return (head
            + f"<p class='mute'>Область измерения — стык двух отростков, образующих сустав "
              f"(нижний отросток верхнего позвонка и верхний отросток нижнего). Порог яркости: "
              f"{data.get('threshold_value','—')} от "
              f"{html.escape(str(data.get('threshold_reference')))}.</p>"
            + "<table><tr><th>Сустав</th><th>Объём слева, мм³</th><th>Объём справа, мм³</th>"
              "<th>Доля ярких слева</th><th>Доля ярких справа</th><th>Отношение</th>"
              f"<th>Больше</th></tr>{''.join(rows)}</table>"
            + top_html
            + f"<div class='warn'><b>Границы применимости:</b><ul>{limits}</ul></div>")


def _reliability_section(ctx: Context) -> str:
    """Alignment and per-level model agreement — the caveats behind every number."""
    blocks = []

    reg = ctx.stage_data("register")
    for name, target in sorted((reg.get("targets") or {}).items()):
        r = target.get("registration") or {}
        applied = target.get("applied")
        label = {"fatsat": "серию с подавлением жира", "axial": "аксиальную серию"}.get(name, name)
        blocks.append(
            f"<h3>Совмещение: маски → {label} {badge_html(Evidence.MEASUREMENT)}</h3>"
            f"<p>Поправка на движение между сериями: сдвиг "
            f"<b>{r.get('translation_magnitude_mm','—')} мм</b>, поворот "
            f"<b>{r.get('rotation_deg','—')}°</b>. Применена: <b>"
            f"{'да' if applied else 'нет'}</b>.</p>"
            + ("" if applied else
               f"<div class='warn'>{html.escape(str(r.get('reason') or 'не применена'))} — "
               "маски стоят там, где их поместила геометрия DICOM. Разницу меньше "
               "нескольких миллиметров считать шумом совмещения.</div>")
        )

    cross = ctx.stage_data("crosscheck")
    if cross:
        rows = "".join(
            f"<tr><td>{html.escape(r['level'])}</td><td>{r.get('dice','—')}</td>"
            f"<td>{'да' if r.get('reliable') else 'нет'}</td></tr>"
            for r in cross.get("levels", [])
        )
        weak = cross.get("levels_needing_visual_check") or []
        blocks.append(
            f"<h3>Согласие двух независимых моделей по уровням {badge_html(Evidence.MODEL)}</h3>"
            f"<p class='mute'>Вторая модель: {html.escape(str(cross.get('second_model')))}. "
            f"Средний Dice: {cross.get('mean_dice','—')}.</p>"
            f"<table><tr><th>Уровень</th><th>Dice</th><th>Надёжно</th></tr>{rows}</table>"
            + (f"<div class='warn'>Смотреть глазами перед доверием числам: "
               f"{', '.join(html.escape(w) for w in weak)}.</div>" if weak else
               "<div class='box'>Модели согласуются на всех уровнях.</div>")
        )

    mirror = (ctx.stage_data("spineps") or {}).get("mirror_consistency") or {}
    if mirror.get("tested"):
        agreement = mirror.get("side_label_agreement") or {}
        stable = mirror.get("sides_stable")
        blocks.append(
            f"<h3>Устойчивость определения стороны {badge_html(Evidence.MODEL)}</h3>"
            f"<p>Модель прогнана повторно на зеркально отражённой копии; метки сторон "
            f"возвращены обратно и сравнены. Минимальный Dice по меткам сторон: "
            f"<b>{agreement.get('min_dice','—')}</b> (слабейшая метка: "
            f"{html.escape(str(agreement.get('weakest_label')))}).</p>"
            + ("<div class='box'>Определение лево/право устойчиво — сравнения сторон "
               "выше можно читать.</div>" if stable else
               "<div class='warn'>Определение лево/право <b>неустойчиво</b> на этом объёме. "
               "Ни одно сравнение сторон в этом прогоне не следует считать надёжным.</div>")
        )

    if not blocks:
        return ""
    return f"{_h2('Надёжность: совмещение и согласие моделей')}" + "".join(blocks)


def _not_assessable_section(ctx: Context, picks: dict) -> str:
    """The most important section: what this study cannot answer at all."""
    items = [
        "<b>Фасеточный синдром как диагноз.</b> Он ставится клинически; референсным "
        "тестом считается диагностическая блокада медиальной ветви, а не МРТ. "
        "Изменения на МРТ (выпот, отёк, гипертрофия) поддерживают версию, но их "
        "отсутствие её не исключает и наоборот.",
        "<b>Отёк костного мозга и периартикулярный отёк</b> без последовательности с "
        "подавлением жира (STIR/TIRM/SPAIR или Dixon) не оцениваются в принципе.",
        "<b>Грудные фасеточные суставы</b> ориентированы почти во фронтальной плоскости: "
        "по сагиттальным срезам они видны плохо. Нужны аксиальные (или косые) срезы, "
        "лучше с подавлением жира.",
        "<b>Поясничный отдел</b> в этом исследовании отсутствует, если сканирован только "
        "грудной: боль в области грудо-поясничного перехода часто исходит из L1-L5.",
        "<b>Динамика.</b> МРТ лежа не показывает то, что болит при наклоне; "
        "функциональные причины на статичном снимке не видны.",
        "<b>Жировая инфильтрация мышц</b> количественно требует Dixon-последовательности.",
    ]
    if not picks.get("T2_AX"):
        items.append("В этом исследовании <b>нет аксиальной T2</b> — см. пункт про плоскость.")
    return (f"{_h2('Чего эти данные не могут показать')}<div class='warn'><ul>"
            + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>")


def _status_section(ctx: Context) -> str:
    rows = []
    for name, res in ctx.results.items():
        status = res.status.value if hasattr(res.status, "value") else str(res.status)
        rows.append(f"<tr><td><code>{html.escape(name)}</code></td>"
                    f"<td>{html.escape(status)}</td>"
                    f"<td>{html.escape(res.reason or '—')}</td></tr>")
    return (f"{_h2('Что выполнялось')}<table><tr><th>Этап</th><th>Статус</th>"
            f"<th>Причина / примечание</th></tr>{''.join(rows)}</table>")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _stage(ctx: Context, name: str) -> tuple[dict, str | None]:
    """Stage data plus a ready-made 'did not run' block when it is unusable."""
    data = ctx.stage_data(name)
    result = ctx.results.get(name)
    if data:
        return data, None
    reason = (result.reason if result else None) or "этап не выполнялся"
    status = (result.status.value if result and hasattr(result.status, "value") else "n/a")
    return {}, (f"<div class='warn'>Этап не дал результата (<code>{html.escape(status)}</code>): "
                f"{html.escape(str(reason))}</div>")


def _findings_bundle(ctx: Context) -> dict:
    """Machine-readable bundle: every stage's data, with its evidence level."""
    return {
        "subject_id": ctx.config.subject_id,
        "disclaimer": DISCLAIMER_RU,
        "stages": {
            name: {
                "status": res.status.value if hasattr(res.status, "value") else str(res.status),
                "evidence": res.evidence.value if hasattr(res.evidence, "value") else str(res.evidence),
                "reason": res.reason,
                "data": res.data,
            }
            for name, res in ctx.results.items()
        },
    }


def _try_figure(ctx: Context) -> str | None:
    """Mid-sagittal image with the vertebra outlines, if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from ..utils import load_canonical, resample_mask_to
    except Exception:
        return None

    picks = ctx.stage_data("ingest").get("picks", {})
    image_path = picks.get("T2_SAG")
    masks = ctx.stage_data("spineps").get("instance_masks") or []
    if not image_path or not masks:
        return None
    try:
        img = load_canonical(image_path)
        data = np.asarray(img.get_fdata(), dtype=float)
        inst = np.asarray(resample_mask_to(load_canonical(masks[0]), img).get_fdata())
        counts = (inst > 0).sum(axis=(1, 2))
        idx = int(np.argmax(counts))
        zooms = img.header.get_zooms()
        aspect = float(zooms[2]) / float(zooms[1]) if len(zooms) >= 3 else 1.0

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.imshow(np.rot90(data[idx]), cmap="gray", aspect=aspect)
        ax.contour(np.rot90(inst[idx] > 0), levels=[0.5], colors="#22d3ee", linewidths=0.8)
        ax.set_title("Средний сагиттальный срез с контурами сегментации", fontsize=10)
        ax.axis("off")
        out = Path(ctx.config.figures_dir) / "midsagittal_overlay.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(out.read_bytes()).decode("ascii")
    except Exception:
        return None
    return ("<h2>Контроль сегментации</h2>"
            "<p class='mute'>Смотреть в первую очередь: если контуры не совпадают с "
            "позвонками, все числа ниже недействительны.</p>"
            f"<img alt='overlay' src='data:image/png;base64,{b64}'>")
