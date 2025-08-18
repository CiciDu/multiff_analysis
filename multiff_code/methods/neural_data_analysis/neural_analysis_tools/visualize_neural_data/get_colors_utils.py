import re
import colorsys
import plotly.colors as pc

try:
    # nice-to-have: resolves CSS names to RGB
    from matplotlib.colors import to_rgb as _mpl_to_rgb  # returns floats 0..1
except Exception:
    _mpl_to_rgb = None

def _rgb01_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(
        int(round(max(0, min(1, r)) * 255)),
        int(round(max(0, min(1, g)) * 255)),
        int(round(max(0, min(1, b)) * 255)),
    )

def _parse_color_to_rgb01(c):
    """Return (r,g,b) floats in 0..1 from hex, css name, rgb(), or hsl()."""
    if isinstance(c, (tuple, list)) and len(c) == 3:
        r, g, b = c
        # assume 0..1, but auto-scale if looks like 0..255
        if max(r, g, b) > 1:
            r, g, b = r/255.0, g/255.0, b/255.0
        return float(r), float(g), float(b)

    if not isinstance(c, str):
        raise ValueError(f"Unsupported color type: {type(c)}")

    s = c.strip().lower()

    # hex: #rgb or #rrggbb
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(ch*2 for ch in h)
        if len(h) != 6 or not re.fullmatch(r"[0-9a-f]{6}", h):
            raise ValueError(f"Invalid hex color: {c}")
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return r, g, b

    # rgb(r,g,b)
    m = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if m:
        r, g, b = (int(m.group(i)) / 255.0 for i in (1, 2, 3))
        return r, g, b

    # hsl(h, s%, l%)
    m = re.fullmatch(r"hsl\(\s*([0-9.]+)\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)", s)
    if m:
        h = float(m.group(1)) % 360.0
        S = float(m.group(2)) / 100.0
        L = float(m.group(3)) / 100.0
        # colorsys uses HLS (L in middle)
        r, g, b = colorsys.hls_to_rgb(h/360.0, L, S)
        return r, g, b

    # CSS color names (e.g., "yellow", "rebeccapurple")
    if _mpl_to_rgb is not None:
        try:
            r, g, b = _mpl_to_rgb(s)
            return r, g, b
        except ValueError:
            pass

    raise ValueError(f"Unrecognized color format: {c}")

def _normalize_color(c, min_light=0.25, max_light=0.45, min_sat=0.55):
    """Clamp lightness/saturation so colors aren’t too light/washed out."""
    r, g, b = _parse_color_to_rgb01(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # HLS order
    l = max(min(l, max_light), min_light)
    s = max(s, min_sat)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r2, g2, b2)



def _get_colors(num_colors):
    base = [_normalize_color(c) for c in pc.qualitative.Set3]
    if num_colors <= len(base):
        return base[:num_colors]
    else:
        # fallback: evenly spaced HSL, already in a safe lightness/saturation
        return [ _normalize_color(f"hsl({i*360/num_colors}, 75%, 45%)")
                 for i in range(num_colors) ]
