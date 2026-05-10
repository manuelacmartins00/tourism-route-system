"""
heatmap_pois.py
Gera um heatmap interactivo Folium dos POIs por densidade geográfica.
Uso: python heatmap_pois.py
Output: heatmap_pois.html (abre no browser)
"""

import json
import folium
from folium.plugins import HeatMap, FastMarkerCluster
from collections import Counter
from pathlib import Path

# ── Configuração ──────────────────────────────────────────────
DATA_FILE = "data/portugal_todos_pois_final_enriched.json"
OUTPUT    = "heatmap_pois.html"
TOP_N     = 15  # número de zonas com mais POIs a listar no terminal

# ── Carregar dados ────────────────────────────────────────────
print(f"A carregar {DATA_FILE}...")
data = json.loads(Path(DATA_FILE).read_text(encoding="utf-8"))
pois = data if isinstance(data, list) else data.get("pois", [])
print(f"  Total POIs: {len(pois)}")

# ── Cores por categoria ───────────────────────────────────────
CAT_COLORS = {
    "museus_e_palacios":    "#4f8ef7",
    "monumentos":           "#a78bfa",
    "restaurantes_e_cafes": "#f87171",
    "bares_e_discotecas":   "#fb923c",
    "espacos_verdes":       "#34d399",
    "parques_e_reservas":   "#6ee7b7",
    "praias":               "#fcd34d",
    "turismo_activo":       "#f472b6",
    "arqueologia":          "#c084fc",
    "eventos":              "#38bdf8",
}
DEFAULT_COLOR = "#94a3b8"

# ── Extrair coordenadas e metadados ──────────────────────────
coords   = []
markers  = []   # (lat, lon, name, category)
by_bundle = Counter()

for p in pois:
    loc = p.get("location", {})
    lat = loc.get("lat") or p.get("lat")
    lon = loc.get("lon") or p.get("lon")
    if lat and lon and -90 <= lat <= 90 and -180 <= lon <= 180:
        coords.append([lat, lon])
        cat  = p.get("category", p.get("source", {}).get("bundle", "?"))
        name = p.get("name", "—")
        markers.append((lat, lon, name, cat))
        by_bundle[cat] += 1

print(f"  POIs com coordenadas válidas: {len(coords)}")

# ── Gerar heatmap ─────────────────────────────────────────────
m = folium.Map(
    location=[39.5, -8.0],  # centro de Portugal
    zoom_start=7,
    tiles="CartoDB dark_matter"
)

HeatMap(
    coords,
    radius=18,
    blur=15,
    max_zoom=13,
    gradient={0.2: "#1a1d27", 0.4: "#2a4a7f", 0.6: "#4f8ef7", 0.8: "#34d399", 1.0: "#f87171"},
    name="Heatmap",
).add_to(m)

# ── Marcadores individuais (visíveis ao fazer zoom in) ────────
callback = """
function(row) {
    var color  = row[4] || '#94a3b8';
    var marker = L.circleMarker([row[0], row[1]], {
        radius: 6,
        fillColor: color,
        color: '#fff',
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.85
    });
    marker.bindPopup(
        '<b>' + row[2] + '</b><br/><span style="font-size:11px;color:#666">' + row[3] + '</span>'
    );
    marker.bindTooltip(row[2], {direction:'top', offset:[0,-4]});
    return marker;
}
"""

marker_data = [
    [lat, lon, name, cat, CAT_COLORS.get(cat, DEFAULT_COLOR)]
    for lat, lon, name, cat in markers
]

cluster = FastMarkerCluster(
    marker_data,
    callback=callback,
    name="POIs individuais",
    show=False,          # começa escondido — aparece automaticamente ao zoom in
).add_to(m)

# Layer control + JS para mostrar/esconder automaticamente por zoom
folium.LayerControl(collapsed=False).add_to(m)

zoom_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var maps = Object.values(window).filter(v => v && v._leaflet_id);
    var map  = maps.find(v => v.getZoom);
    if (!map) return;
    map.on('zoomend', function() {
        var z = map.getZoom();
        map.eachLayer(function(layer) {
            if (layer.options && layer.options.name === 'POIs individuais') {
                if (z >= 12) map.addLayer(layer);
                else         map.removeLayer(layer);
            }
        });
    });
});
</script>
"""
m.get_root().html.add_child(folium.Element(zoom_js))

# Adicionar marcadores das capitais de distrito para referência
capitais = {
    "Lisboa": (38.717, -9.139),
    "Porto": (41.157, -8.629),
    "Coimbra": (40.203, -8.410),
    "Braga": (41.545, -8.426),
    "Faro": (37.019, -7.935),
    "Évora": (38.571, -7.909),
    "Aveiro": (40.644, -8.645),
    "Viseu": (40.657, -7.909),
    "Leiria": (39.744, -8.807),
    "Setúbal": (38.524, -8.893),
    "Santarém": (39.236, -8.686),
    "Viana do Castelo": (41.694, -8.834),
    "Vila Real": (41.301, -7.745),
    "Bragança": (41.806, -6.757),
    "Guarda": (40.538, -7.268),
    "Castelo Branco": (39.820, -7.491),
    "Portalegre": (39.296, -7.428),
    "Beja": (38.015, -7.863),
}

for nome, (lat, lon) in capitais.items():
    folium.Marker(
        location=[lat, lon],
        tooltip=nome,
        icon=folium.DivIcon(html=f"""
            <div style="
                font-size:10px;font-weight:600;color:white;
                background:rgba(0,0,0,0.6);
                padding:2px 5px;border-radius:4px;
                white-space:nowrap;
            ">{nome}</div>
        """)
    ).add_to(m)

m.save(OUTPUT)
print(f"\nHeatmap guardado: {OUTPUT}")

# ── Top bundles ───────────────────────────────────────────────
print(f"\nTop {TOP_N} categorias com mais POIs:")
for bundle, count in by_bundle.most_common(TOP_N):
    bar = "█" * (count // 50)
    print(f"  {bundle:<35} {count:>5}  {bar}")

# ── Divisão por região aproximada (para envio aos postos) ─────
print("\nSugestao de postos de turismo a contactar (por densidade):")
regioes = {
    "Porto e Norte":      lambda lat, lon: lat > 41.0,
    "Centro":             lambda lat, lon: 39.5 < lat <= 41.0,
    "Lisboa e Vale do Tejo": lambda lat, lon: 38.3 < lat <= 39.5,
    "Alentejo":           lambda lat, lon: 37.5 < lat <= 38.3,
    "Algarve":            lambda lat, lon: lat <= 37.5 and lon > -9.5,
}
regiao_counts = Counter()
for lat, lon in coords:
    for nome, fn in regioes.items():
        if fn(lat, lon):
            regiao_counts[nome] += 1
            break
    else:
        regiao_counts["Ilhas"] += 1

for regiao, count in regiao_counts.most_common():
    print(f"  {regiao:<30} {count:>5} POIs")