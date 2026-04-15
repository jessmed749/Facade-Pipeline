"""
osm_height.py — Query OpenStreetMap for building height using GPS coordinates.

Replaces the manual KNOWN_WALL_HEIGHT_M constant with an automatic lookup
against the Overpass API.  Falls back to the manual value if OSM has no data.

Usage:

    from osm_height import lookup_building_height

    height_m = lookup_building_height(
        lat=30.288753,
        lon=-97.736348,
        fallback_m=28.0,
        search_radius_m=30,
    )
"""

import urllib.request
import urllib.parse
import json
import math

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Levels × this = estimated height when only 'building:levels' is tagged (no 'height')
METERS_PER_LEVEL = 3.5


def lookup_building_height(
    lat: float,
    lon: float,
    fallback_m: float = 28.0,
    search_radius_m: int = 30,
    timeout_s: int = 10,
) -> tuple[float, str]:
    """
    Query Overpass for the building at (lat, lon) and return its height.

    Returns
    -------
    height_m : float
    source   : str — one of "osm:height", "osm:levels", "fallback"
    """
    print(f"OSM height lookup: ({lat:.6f}, {lon:.6f})  radius={search_radius_m} m …")

    query = f"""
[out:json][timeout:{timeout_s}];
(
  way["building"](around:{search_radius_m},{lat},{lon});
  relation["building"](around:{search_radius_m},{lat},{lon});
);
out tags;
"""
    try:
        data = urllib.parse.urlencode({"data": query}).encode()
        req  = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
        req.add_header("User-Agent", "facade-sam-pipeline/1.0")

        with urllib.request.urlopen(req, timeout=timeout_s + 5) as resp:
            result = json.loads(resp.read().decode())

        elements = result.get("elements", [])
        print(f"  OSM returned {len(elements)} building element(s)")

        for el in elements:
            tags = el.get("tags", {})

            # Prefer explicit 'height' tag (in metres)
            if "height" in tags:
                try:
                    raw = tags["height"].replace("m", "").strip()
                    h = float(raw)
                    print(f"  Found 'height' tag: {h} m")
                    return h, "osm:height"
                except ValueError:
                    pass

            # Fall back to building:levels * 3.5 m
            if "building:levels" in tags:
                try:
                    levels = float(tags["building:levels"])
                    h = levels * METERS_PER_LEVEL
                    print(f"  Found 'building:levels': {levels} → {h:.1f} m")
                    return h, "osm:levels"
                except ValueError:
                    pass

        print(f"  No height data in OSM — using fallback {fallback_m} m")
        return fallback_m, "fallback"

    except Exception as e:
        print(f"  OSM lookup failed ({e}) — using fallback {fallback_m} m")
        return fallback_m, "fallback"