# ChatGPT5 assistend code
# save as csv_table_diff.py
import pandas as pd
from pathlib import Path
import html
import re


# --- EDIT THESE PATHS / KEY ---
OLD = r"C:\Users\chyi\draftsh2025\temp_devs\merged_dataset_0917.csv"
NEW = r"C:\Users\chyi\draftsh2025\temp_devs\merged_dataset_forward.csv"
KEY_COL = "index_0810"   # your unique key column
OUT_PATH = "temp_devs\\0917to0918.html"

# ---------- load & normalize ----------
old: pd.DataFrame = pd.read_csv(OLD, dtype=str, keep_default_na=False).applymap(str.strip)
new: pd.DataFrame = pd.read_csv(NEW, dtype=str, keep_default_na=False).applymap(str.strip)

# ensure key exists
if KEY_COL not in old.columns:
    old[KEY_COL] = range(len(old))
if KEY_COL not in new.columns:
    new[KEY_COL] = range(len(new))

# wish-list columns to keep near the front when available
keep_wishlist = ["idx_0917", "elements", "elements_fraction"]
keep_cols = [KEY_COL] + [c for c in keep_wishlist if c in old.columns and c in new.columns]
_seen = set(); keep_cols = [c for c in keep_cols if not (c in _seen or _seen.add(c))]

# shared/non-key columns
common_cols = [c for c in old.columns if c in new.columns]
nonkey_cols  = [c for c in common_cols if c != KEY_COL]

# choose order column (for ranges and sorting)
ORDER_COL = "idx_0810" if "idx_0810" in common_cols else ("index_0810" if "index_0810" in common_cols else KEY_COL)
DISPLAY_ORDER = ORDER_COL  # used in titles

# reindex by key
old_k = old.set_index(KEY_COL, drop=False)
new_k = new.set_index(KEY_COL, drop=False)

# ---------- added / removed / changed ----------
added_keys   = new_k.index.difference(old_k.index)
removed_keys = old_k.index.difference(new_k.index)
common_keys  = old_k.index.intersection(new_k.index)

old_common = old_k.loc[common_keys, nonkey_cols]
new_common = new_k.loc[common_keys, nonkey_cols]
diff_mask  = (old_common != new_common)

# map each key -> exact set of changed columns (frozenset)
changed_sets = {k: [c for c in nonkey_cols if diff_mask.loc[k, c]] for k in common_keys}
changed_sets = {k: s for k, s in changed_sets.items() if len(s) > 0}

# ---------- helpers ----------
def table_css(sty):
    # compact tables; wrap text; consistent look
    return (sty.hide(axis="index")
              .set_table_attributes("class='mini'")
              .set_properties(subset=pd.IndexSlice[:, :], **{
                  "max-width": "640px",
                  "white-space": "normal",
                  "word-break": "break-word",
              }))

def add_break(html_snip: str) -> str:
    return html_snip + "<br/>"

def to_html_table(df, caption, row_color=None, anchor=None):
    if df.empty: 
        return ""
    cap = html.escape(caption)
    sty = df.style.set_caption(cap)
    sty = table_css(sty)
    if row_color:
        sty = sty.apply(lambda r: [f"background-color:{row_color}"]*len(r), axis=1)
    out = sty.to_html()
    if anchor:
        out = f"<a id='{anchor}'></a>" + out
    return add_break(out)

def _num(s):
    # numeric if possible, else NaN; used for sorting / ranges
    return pd.to_numeric(s, errors="coerce")

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# ---------- compact Added/Removed ----------
added_tbl = to_html_table(
    new_k.loc[added_keys, [c for c in keep_cols if c != KEY_COL]].reset_index().sort_values(by=[DISPLAY_ORDER] if DISPLAY_ORDER in new_k.columns else [KEY_COL], key=_num),
    "Added rows (in NEW only)", row_color="#e6ffed", anchor="added"
)
removed_tbl = to_html_table(
    old_k.loc[removed_keys, [c for c in keep_cols if c != KEY_COL]].reset_index().sort_values(by=[DISPLAY_ORDER] if DISPLAY_ORDER in old_k.columns else [KEY_COL], key=_num),
    "Removed rows (in OLD only)", row_color="#ffeef0", anchor="removed"
)

# ---------- group by exact changed-set, then split into contiguous ranges ----------
def contiguous_ranges(keys):
    """
    Given iterable of keys, return list of (start, end, ordered_keys)
    where start/end are from DISPLAY_ORDER and contiguity is diff==1 on that column.
    """
    # build small df with key + order value (prefer old then new)
    base_cols = []
    for c in (DISPLAY_ORDER, KEY_COL):
        if c in old_k.columns and c not in base_cols:
            base_cols.append(c)
    df_subset = old_k.loc[keys, base_cols].copy()

    # if order column missing in old, pull it from new
    if DISPLAY_ORDER not in df_subset.columns and DISPLAY_ORDER in new_k.columns:
        df_subset[DISPLAY_ORDER] = new_k.loc[keys, DISPLAY_ORDER]

    # sort ascending by DISPLAY_ORDER (numeric if possible), then by KEY_COL
    if DISPLAY_ORDER in df_subset.columns:
        df_subset["_ord"] = _num(df_subset[DISPLAY_ORDER])
        df_subset.index.name="idx"
        df_subset = df_subset.sort_values(["_ord", KEY_COL], kind="mergesort")
    else:
        df_subset = df_subset.sort_values(KEY_COL, kind="mergesort")

    # build contiguous groups
    if DISPLAY_ORDER in df_subset.columns and df_subset["_ord"].notna().all():
        diffs = df_subset["_ord"].diff().fillna(1)
        grp_id = (diffs != 1).cumsum()
        groups = []
        for _, part in df_subset.groupby(grp_id, sort=False):
            start = int(part["_ord"].iloc[0])
            end   = int(part["_ord"].iloc[-1])
            groups.append((start, end, part[KEY_COL].tolist()))
        return groups
    else:
        # fallback: each row is its own range
        return [(None, None, [k]) for k in df_subset[KEY_COL].tolist()]

def group_table_for_set(cols_changed: tuple, keys_for_set):
    """
    Build one table per contiguous range for this exact set of changed columns.
    Returns list of (anchor_id, title, html_table)
    """
    ranges = contiguous_ranges(keys_for_set)
    pretty_label = ", ".join(f"`{c}`" for c in cols_changed) if cols_changed else "(none)"
    # for summary we also need [start:end] text and anchors
    blocks = []
    for start, end, key_list in ranges:
        # assemble table rows (sorted by DISPLAY_ORDER asc)
        rows = []
        # fetch order values to sort and display
        ord_vals = []
        for k in key_list:
            ov = old_k.at[k, DISPLAY_ORDER] if DISPLAY_ORDER in old_k.columns else (new_k.at[k, DISPLAY_ORDER] if DISPLAY_ORDER in new_k.columns else "")
            ord_vals.append(_num(ov))
        # sort key_list by numeric order, stable
        key_list = [k for _, k in sorted(zip(ord_vals, key_list), key=lambda t: (t[0],))]

        for k in key_list:
            row = {DISPLAY_ORDER: old_k.at[k, DISPLAY_ORDER] if DISPLAY_ORDER in old_k.columns else (new_k.at[k, DISPLAY_ORDER] if DISPLAY_ORDER in new_k.columns else ""),
                   KEY_COL: k}
            for c in cols_changed:
                row[f"{c} (OLD)"] = old_k.at[k, c]
                row[f"{c} (NEW)"] = new_k.at[k, c]
            rows.append(row)

        df_g = pd.DataFrame(rows)

        # color: OLD red, NEW yellow
        def color_cols(_, colnames=df_g.columns):
            colors = []
            for c in colnames:
                if c.endswith("(OLD)"):
                    colors.append("background-color:#ffe1e1")
                elif c.endswith("(NEW)"):
                    colors.append("background-color:#fff5b1")
                else:
                    colors.append("")
            return colors

        # title + anchor
        if start is not None and end is not None:
            title = f"{', '.join(cols_changed)} changed [{start}:{end}] ({DISPLAY_ORDER})"
            anchor = f"group_{sanitize('_'.join(cols_changed))}_{start}_{end}"
        else:
            first = str(df_g[DISPLAY_ORDER].iloc[0]) if DISPLAY_ORDER in df_g.columns else ""
            title = f"{', '.join(cols_changed)} change at {first} ({DISPLAY_ORDER})"
            anchor = f"group_{sanitize('_'.join(cols_changed))}_{sanitize(first)}"

        sty = df_g.style.set_caption(title)
        sty = table_css(sty).apply(color_cols, axis=1)
        tbl = add_break(f"<a id='{anchor}'></a>" + sty.to_html())
        blocks.append((anchor, start, end, title, tbl))
    return blocks

# Build groups for EVERY exact changed-set
from collections import defaultdict
set_to_keys = defaultdict(list)
for k, s in changed_sets.items():
    set_to_keys[tuple(sorted(s))].append(k)

group_blocks = []
for cols_changed, keys_for_set in sorted(set_to_keys.items(), key=lambda kv: (len(kv[0]), kv[0])):  # stable order
    pass

cols_changed=None
keys_for_set=[]
for i, s in changed_sets.items():
    if cols_changed is None:
        cols_changed=s

    if cols_changed!=s:
        group_blocks.extend(group_table_for_set(cols_changed, keys_for_set))
        cols_changed=s
        keys_for_set=[i]
    elif cols_changed==s:
        keys_for_set.append(i)

# ---------- HTML summary ----------
summary_lines = [
    f"<b>Summary</b>",
    f"Added: <a href='#added'>{len(added_keys)}</a> &nbsp;|&nbsp; "
    f"Removed: <a href='#removed'>{len(removed_keys)}</a> &nbsp;|&nbsp; "
    f"Changed rows: {sum(len(v) for v in set_to_keys.values())}",
    "<br/>"
]

# collect per-set ranges for link-y summary (e.g., `short_cite` changed: [0:87], ...)
per_set_ranges = defaultdict(list)
for anchor, start, end, _, _ in group_blocks:
    # find set label from anchor
    m = re.match(r"group_(.+?)_(?:\d*_\d*\b)", anchor)
    key = m.group(1) if m else anchor
    per_set_ranges[key].append((start, end, anchor))

def backtick_split(key):
    # convert anchor key like short_cite__elements_fraction into backticked names
    cols=[]
    for col in common_cols:
        if col in key:
            if col=="elements":
                if "elements_fraction" in key:
                    pass
                else: 
                    cols.append(col)
            else:
                cols.append(col)
    return ", ".join(f"`{c}`" for c in cols)

# rebuild mapping using readable set names
pretty_map = defaultdict(list)
for anchor_key, lst in per_set_ranges.items():
    for start, end, anc in lst:
        if start is not None and end is not None:
            pretty_map[anchor_key].append((f"[{start}:{end}]", anc))
        else:
            pretty_map[anchor_key].append((f"[{DISPLAY_ORDER}]", anc))

if pretty_map:
    summary_lines.append("<b>Grouped changes</b><ul>")
    # Present per changed set in descending frequency of rows (optional) or name; keep name.
    for anchor_key in sorted(pretty_map.keys()):
        label = backtick_split(anchor_key)
        links = ", ".join([f"<a href='#{anc}'>{rng}</a>" for rng, anc in pretty_map[anchor_key]])
        summary_lines.append(f"<li>{label} changed: {links}</li>")
    summary_lines.append("</ul>")

# ---------- assemble HTML ----------
header = f"""<!doctype html>
<meta charset="utf-8"><title>CSV Table Diff</title>
<style>
caption{{caption-side:top;font-weight:600;margin:10px 0}}
table.mini{{border-collapse:collapse;margin:10px 0;display:block;max-width:100%;table-layout:auto;overflow:auto}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f6f6f6}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:12px}}
a{{text-decoration:none}}
</style>
<div>{''.join(summary_lines)}</div>
<br/>
"""

parts = [header, added_tbl, removed_tbl]

if group_blocks:
    parts.append("<div class='section-title'><b>Grouped changes (exact column sets)</b></div><br/>")
    for _, _, _, _, tbl in group_blocks:
        parts.append(tbl)
else:
    parts.append("<p>No changed rows.</p><br/>")

html_out = "".join(parts)
Path(OUT_PATH).write_text(html_out, encoding="utf-8")
print(f"Wrote {OUT_PATH}")
