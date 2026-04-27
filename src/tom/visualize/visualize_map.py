#!/usr/bin/env python3
"""
Generate interactive polygon map from poi_name_authority + polygon_classification.
Uses the finalized 3-case classification: OWNED / SHARED_DISTINCT / SHARED_BUILDING.

Two overlay layers: 2021 (stable+closed) and 2022 (stable+new+unknown).
Polygon color encodes classification type; border color encodes temporal status.
Click for full details.
"""

import json
import os
import pickle
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping, Point

BASE = "/scratch/sx2490/econai/nyc_metro"
AUTH_PATH = f"{BASE}/data/poi_name_authority.csv"
PKL_PATH  = f"{BASE}/data/polygon_spatial_index.pkl"
OUTPUT    = f"{BASE}/poi_polygon_map.html"

# Fill colors by polygon type
FILL = {
    "OWNED":           "#2196F3",   # blue
    "SHARED_DISTINCT": "#00BCD4",   # teal
    "SHARED_BUILDING": "#FF9800",   # orange
    "FALLBACK":        "#9E9E9E",   # gray (no polygon, circle marker)
}
# Border colors by temporal status
BORDER = {
    "stable":              "#1565C0",   # dark blue
    "new":                 "#2E7D32",   # dark green
    "closed_2022":         "#E65100",   # dark orange
    "closed_before_study": "#B71C1C",   # dark red
    "closed":              "#B71C1C",   # dark red (WPP-only closed)
    "unknown":             "#757575",   # gray
}


def main():
    print("Loading data...")
    with open(PKL_PATH, "rb") as f:
        idx = pickle.load(f)
    polygons = idx["polygons"]
    fallback_pois = idx["fallback_pois"]
    print(f"  {len(polygons):,} polygons, {len(fallback_pois):,} fallback POIs")

    # Build GeoJSON features for a temporal filter set
    def build_features(temporal_filter):
        feats = []
        for p in polygons:
            matching = [m for m in p["members"] if m["temporal_status"] in temporal_filter]
            if not matching:
                continue
            geom_json = mapping(p["geometry"])
            ptype = p["polygon_type"]
            if ptype in ("OWNED", "SHARED_DISTINCT"):
                for m in matching:
                    feats.append({"type": "Feature", "geometry": geom_json, "properties": {
                        "n": m["name"], "pt": ptype, "ts": m["temporal_status"],
                        "fc": FILL[ptype], "bc": BORDER.get(m["temporal_status"], "#757575"),
                        "pk": m.get("PLACEKEY", ""), "cat": m.get("sub_category", ""),
                        "np": p["n_pois"], "v": int(m.get("sg21_visits", 0)),
                        "src": m.get("name_source", ""), "mn": "",
                    }})
            else:
                member_names = [m["name"] for m in p["members"][:8]]
                ts = matching[0]["temporal_status"]
                feats.append({"type": "Feature", "geometry": geom_json, "properties": {
                    "n": p["building_name"] or member_names[0], "pt": ptype, "ts": ts,
                    "fc": FILL[ptype], "bc": BORDER.get(ts, "#757575"),
                    "pk": "", "cat": "Food Court / Complex", "np": p["n_pois"],
                    "v": sum(int(m.get("sg21_visits", 0)) for m in p["members"]),
                    "src": "", "mn": " | ".join(member_names),
                }})
        # Fallback (no polygon)
        for p in fallback_pois:
            if p["temporal_status"] not in temporal_filter:
                continue
            feats.append({"type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": {
                    "n": p["name"], "pt": "FALLBACK", "ts": p["temporal_status"],
                    "fc": FILL["FALLBACK"], "bc": BORDER.get(p["temporal_status"], "#757575"),
                    "pk": p.get("PLACEKEY", ""), "cat": p.get("sub_category", ""),
                    "np": 0, "v": 0, "src": "", "mn": "",
                }})
        return feats

    # Build separate layers
    layers = {
        "stable":  {"filter": {"stable"}, "label": "Stable (both years)"},
        "closed_2022": {"filter": {"closed_2022"}, "label": "Closed in 2022"},
        "closed_before": {"filter": {"closed_before_study"}, "label": "Closed before study"},
        "new":     {"filter": {"new"}, "label": "New (2022 only)"},
    }

    layer_geojsons = {}
    for key, cfg in layers.items():
        print(f"Building layer: {cfg['label']}...")
        feats = build_features(cfg["filter"])
        print(f"  {len(feats):,} features")
        layer_geojsons[key] = json.dumps(
            {"type": "FeatureCollection", "features": feats}, ensure_ascii=False)
        cfg["count"] = len(feats)
    total_gj = sum(len(v) for v in layer_geojsons.values())
    print(f"Total GeoJSON size: {total_gj/1024/1024:.1f} MB")

    sc = layers["stable"]["count"]
    c2c = layers["closed_2022"]["count"]
    cbc = layers["closed_before"]["count"]
    nc = layers["new"]["count"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NYC Metro Restaurant POI Polygon Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body{{margin:0}}
#map{{position:absolute;top:0;bottom:0;width:100%}}
.info{{background:white;padding:8px 12px;border-radius:5px;border:1px solid #ccc;
  font:12px/1.5 sans-serif;max-width:340px}}
.info h4{{margin:0 0 4px;font-size:14px}}
.info table{{border-collapse:collapse;width:100%}}
.info td{{padding:1px 4px;vertical-align:top}}
.info td:first-child{{font-weight:bold;color:#555;white-space:nowrap}}
.badge{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;
  font-weight:bold;color:white;margin-left:4px}}
.legend{{background:white;padding:10px 14px;border-radius:5px;border:1px solid #ccc;
  font:12px/1.8 sans-serif}}
.legend b{{font-size:13px}}
.lsq{{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:middle;
  border-radius:2px}}
.lcirc{{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:middle;
  border-radius:50%;border:2px solid}}
</style>
</head>
<body>
<div id="map"></div>
<script>
var map=L.map('map').setView([40.74,-73.96],12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png',{{
  attribution:'&copy; OSM &copy; CARTO',maxZoom:20}}).addTo(map);

var ptNames={{"OWNED":"Single Restaurant","SHARED_DISTINCT":"Shared (Distinct)",
  "SHARED_BUILDING":"Food Court/Complex","FALLBACK":"No Polygon"}};
var tsLabels={{"stable":"STABLE","new":"NEW","closed_2022":"CLOSED 2022",
  "closed_before_study":"CLOSED PRE-2021","closed":"CLOSED"}};
var tsColors={{"stable":"#1565C0","new":"#2E7D32","closed_2022":"#E65100",
  "closed_before_study":"#B71C1C","closed":"#B71C1C"}};

function popup(p){{
  var mn=p.mn?'<tr><td>Members</td><td style="font-size:10px">'+p.mn+'</td></tr>':'';
  var tsC=tsColors[p.ts]||'#757575';
  var tsL=tsLabels[p.ts]||p.ts.toUpperCase();
  return '<div class="info"><h4>'+p.n+
    '<span class="badge" style="background:'+p.fc+'">'+ptNames[p.pt]+'</span></h4>'+
    '<span class="badge" style="background:'+tsC+'">'+tsL+'</span>'+
    '<table>'+
    '<tr><td>Category</td><td>'+p.cat+'</td></tr>'+
    '<tr><td>POIs in polygon</td><td>'+p.np+'</td></tr>'+
    '<tr><td>2021 Visits</td><td>'+(p.v?p.v.toLocaleString():'-')+'</td></tr>'+
    mn+
    (p.pk?'<tr><td>PLACEKEY</td><td style="font-size:10px">'+p.pk+'</td></tr>':'')+
    (p.src?'<tr><td>Name source</td><td>'+p.src+'</td></tr>':'')+
    '</table></div>';
}}

function makeLayer(gj){{
  return L.geoJSON(gj,{{
    style:function(f){{
      var p=f.properties;
      if(f.geometry.type==='Point')return{{radius:4,color:p.bc,fillColor:p.fc,fillOpacity:0.6,weight:1.5}};
      return{{color:p.bc,weight:2,opacity:0.8,fillColor:p.fc,fillOpacity:0.3}};
    }},
    pointToLayer:function(f,ll){{
      return L.circleMarker(ll,{{radius:4,color:f.properties.bc,
        fillColor:f.properties.fc,fillOpacity:0.6,weight:1.5}});
    }},
    onEachFeature:function(f,layer){{
      layer.bindPopup(function(){{return popup(f.properties)}},{{maxWidth:360}});
      layer.bindTooltip(f.properties.n,{{sticky:true,opacity:0.9}});
    }}
  }});
}}

var gjStable={layer_geojsons["stable"]};
var gjClosed2022={layer_geojsons["closed_2022"]};
var gjClosedBefore={layer_geojsons["closed_before"]};
var gjNew={layer_geojsons["new"]};

var lStable=makeLayer(gjStable);
var lClosed2022=makeLayer(gjClosed2022);
var lClosedBefore=makeLayer(gjClosedBefore);
var lNew=makeLayer(gjNew);

lStable.addTo(map);

L.control.layers(null,{{
  '\\u2705 Stable — both years ({sc:,})':lStable,
  '\\u274c Closed in 2022 ({c2c:,})':lClosed2022,
  '\\u26aa Closed before study ({cbc:,})':lClosedBefore,
  '\\u2728 New — 2022 only ({nc:,})':lNew
}},{{collapsed:false}}).addTo(map);

var legend=L.control({{position:'bottomleft'}});
legend.onAdd=function(){{
  var d=L.DomUtil.create('div','legend');
  d.innerHTML='<b>Polygon Type (fill)</b><br>'+
    '<span class="lsq" style="background:#2196F3"></span>Single Restaurant (OWNED)<br>'+
    '<span class="lsq" style="background:#00BCD4"></span>Shared — Distinct Restaurants<br>'+
    '<span class="lsq" style="background:#FF9800"></span>Food Court / Complex<br>'+
    '<span class="lcirc" style="background:#9E9E9E;border-color:#757575"></span>No Polygon (fallback)<br>'+
    '<br><b>Temporal Status (border)</b><br>'+
    '<span class="lsq" style="background:#1565C0"></span>Stable<br>'+
    '<span class="lsq" style="background:#E65100"></span>Closed in 2022<br>'+
    '<span class="lsq" style="background:#B71C1C"></span>Closed before study<br>'+
    '<span class="lsq" style="background:#2E7D32"></span>New (2022 only)<br>'+
    '<br><small>Fill = polygon type &nbsp;|&nbsp; Border = temporal</small>';
  return d;
}};
legend.addTo(map);
</script>
</body>
</html>"""

    print(f"Writing {OUTPUT}...")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"Done! {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
