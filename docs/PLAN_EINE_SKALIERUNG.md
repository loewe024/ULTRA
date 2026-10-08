# Plan: Eine gespeicherte Skalierung pro Objekt

Stand: 2026-10-08. Voraussetzung: Die Objekt-URDFs sind entfernt. Objekte kommen nur noch als OBJ aus InterAct
(`simulation/intermimic/data/assets/objects/<dataset>/<name>/<name>.obj`).

## Ausgangslage

Jedes Objekt liegt heute mehrfach vor:

| Pfad | Zweck |
|---|---|
| `ultra/data/assets/objects/objects/<name>/<name>.obj` | Originalgrösse; MuJoCo (sim2sim, `render_rollout.py`), Punkt-Sampling in sim2sim |
| `ultra/data/assets/objects/diverse/<name>_100_100_100/<name>_100_100_100.obj` | gleiche Geometrie, eigene Kopie |
| `ultra/data/assets/objects/diverse/<name>_080_080_080/<name>_080_080_080.obj` | vorskaliert, Faktor 0.8 um den Ursprung |
| `ultra/data/assets/usd/objects/<name>_<scale>/<name>_<scale>_h{5,10}.usd` | aus `diverse/` konvertiert (8 Objekte × 2 Hüllenbudgets) |

Geprüft an `largebox`: Die Bounding Box von `_080_080_080` ist genau das 0.8-Fache von `_100_100_100`. Die
Variante entsteht also durch reines Skalieren um den Ursprung. Die `_100`-Kopie ist bytegleich mit
`objects/objects/largebox/largebox.obj`.

Die Skalierung ist bereits im Namen der Motion-Datei enthalten (`sub10_largebox_000_080_080_080.pt`).
`ultra/env/tasks/ultra.py:50` bildet daraus den Schlüssel `largebox_080_080_080`.

## Ziel

- Pro Objekt nur noch **ein** OBJ in Originalgrösse: `ultra/data/assets/objects/objects/<name>/<name>.obj`.
- `diverse/` fällt weg.
- Die Skalierung kommt weiterhin aus dem Namenssuffix. Motion-Dateien, Datensätze und Checkpoints behalten ihre
  Namen.
- Die Skalierung wird beim Laden angewendet (trimesh, USD-Spawn, MuJoCo). Beliebige Skalierungen sind dann ohne
  neue Dateien möglich.

## Umsetzung

### 1. Zentraler Helfer `ultra/utils/object_assets.py`

```python
OBJECT_ROOT = "ultra/data/assets/objects/objects"

def parse_object_key(key):          # "largebox_080_080_080" -> ("largebox", (0.8, 0.8, 0.8))
def object_key_from_motion(path):   # Motion-Datei -> "largebox_080_080_080" (Logik aus ultra.py:50)
def object_obj_path(base_name):     # -> objects/<name>/<name>.obj
def load_scaled_mesh(key):          # trimesh.load(...) + apply_scale(scale)
```

`object_from_motion` aus `scripts/render_rollout.py:47` wird durch diesen Helfer ersetzt. Ohne Suffix gilt
Skalierung `(1, 1, 1)`.

### 2. Isaac Lab

- **`ultra/env/tasks/ultra.py`**
  - Z. 50: `object_key_from_motion` verwenden. Die Schlüssel bleiben `<name>_<sx>_<sy>_<sz>` und damit eindeutig
    pro Skalierung, `obj2motion` und `object_id` bleiben unverändert.
  - `_load_target_asset` (Z. 231): `load_scaled_mesh(key)` statt `diverse/<key>/<key>.obj`. `apply_scale` muss
    **vor** OBB, Schwerpunkt und `sample_surface_even` kommen.
- **`ultra/isaac/scene_cfg.py`**
  - `object_usd_path(base_name, max_convex_hulls)` → `usd/objects/<name>/<name>_h{n}.usd`.
  - `build_object_cfg`: pro Schlüssel `UsdFileCfg(usd_path=object_usd_path(base, n), scale=tuple(s * ballSize for s in spec_scale))`.
    `MultiAssetSpawnerCfg` übernimmt die Skalierung pro Asset. Die Reihenfolge der Assets bleibt `object_names`.
- **`scripts/convert_assets.py`** (`convert_objects`): Glob `objects/objects/*/*.obj` statt `objects/diverse/*/*.obj`.
  Das ergibt 4 × 2 statt 8 × 2 USD-Dateien. Den Docstring anpassen.

### 3. Skripte

- `scripts/prepare_retarget_smplx.py` und `scripts/export_retarget_smplx.py`: prüfen
  `object_obj_path(object_name)` statt `diverse/<name>_<scale>/…`.
- `--asset-scale`: Format `^\d{3}_\d{3}_\d{3}$` per Regex validieren statt mit `choices`.
- `scripts/train_retarget_smplx.sh`: `ASSET_SCALE` bleibt als Variable.

### 4. MuJoCo

- `ultra/sim2sim_teacher.py:304` und `ultra/sim2sim_vae.py:464` nehmen heute nur den Basisnamen
  (`split('_')[1]`). Sie laden `<name>.xml` und das Mesh in Originalgrösse, **die Skalierung aus dem Motion-Namen
  wird ignoriert**. Das ist eine bestehende Inkonsistenz zu Isaac Lab und wird dabei behoben:
  - In `merge_mjcf` (bzw. beim Laden der Objekt-XML) `mesh scale` setzen, wie es `render_rollout.py:68` bereits
    tut.
  - `_load_object` sampelt die Punkte mit `load_scaled_mesh(key)`.
- `gxml.py` bleibt unverändert, weil die XMLs in Originalgrösse geschrieben werden.

### 5. Migration

1. Code-Änderungen 1–4.
2. `python scripts/convert_assets.py --force`, danach die alten Ordner `ultra/data/assets/usd/objects/<name>_<scale>/`
   entfernen.
3. `ultra/data/assets/objects/diverse/` löschen.
4. Doku anpassen: `PIPELINE_ABLAUF.md` (Schritt 0, Filter im Export), `PIPELINE_ERKLAERUNG_ISAACLAB.md`
   (Verzeichnisbaum, Abschnitt Asset-Konvertierung „16 USD-Dateien“ → 8).

Bestehende Checkpoints bleiben gültig, wenn die Physik gleich bleibt (siehe Verifikation).

## Risiken

- **Konvexe Zerlegung:** Sie wird dann auf dem unskalierten Mesh berechnet und erst beim Spawn skaliert. Eine
  lineare Abbildung erhält die Konvexität, die Hüllen bleiben also gültig. Weil VHACD mit relativer
  Voxel-Auflösung arbeitet, sollten die Hüllen bei gleichmässiger Skalierung praktisch gleich sein. Bei
  ungleichmässiger Skalierung (z. B. `100_080_100`) können sie sich etwas von einer Zerlegung des vorskalierten
  Meshes unterscheiden.
- **Masse:** PhysX berechnet die Masse aus Dichte × skaliertem Kollisionsvolumen. Erwartet wird das
  0.512-Fache bei 0.8. Das muss mit der heutigen Masse übereinstimmen.
- **Offsets:** `contact_offset` und `rest_offset` sind absolut und hängen nicht von der Skalierung ab, unverändert.
- **`ballSize`:** Es wird mit der Objekt-Skalierung multipliziert und darf nicht doppelt angewendet werden.

## Verifikation

1. **Geometrie (CPU):** Für jedes Objekt und jede Skalierung `load_scaled_mesh(key)` mit dem alten
   `diverse/<key>/<key>.obj` vergleichen: Vertices, `object_points` (Seed 2024) und `object_corners` müssen
   ≤ 1e-5 übereinstimmen.
2. **Physik (Isaac Lab, 1 Env):** Masse, Trägheit und AABB des gespawnten Objekts für alte und neue USD
   vergleichen.
3. **Retargeting:** Kurzer Lauf mit `scripts/train_retarget_smplx.sh` und Checkpoint-Resume. Reward und
   Episodenlänge wie vorher.
4. **Teacher-Inferenz** mit dem mitgelieferten Checkpoint auf einigen `_080`-Clips. Die Tracking-Fehler sollen
   unverändert bleiben.
5. **MuJoCo:** `render_rollout.py` und `scripts/sim2sim_student.sh` mit einem `_080`-Clip. Das Objekt hat die
   richtige Grösse.
