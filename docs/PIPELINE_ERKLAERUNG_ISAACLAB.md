# ULTRA: Aufbau des Repos, Retargeting und Simulation mit Isaac Lab

Diese Erklärung beschreibt den Stand des Branches `feature/isaaclab-port` (Commit `259c34a`), also die Portierung von Isaac Gym auf Isaac Sim 5.1 / Isaac Lab 2.3.2 / rl-games 1.6.1. Die Erklärung der alten Isaac-Gym-Version steht in [`PIPELINE_ERKLAERUNG.md`](PIPELINE_ERKLAERUNG.md).

Grundlage ist das Lesen des Codes, dazu fünf kurze Läufe am 30.09.2026 im Conda-Env `ultra` (RTX 5070 Ti):

- `scripts/check_layout.py` mit 4 Envs: `PASSED`.
- Drei eigene Probeläufe der Retargeting-Umgebung `UltraG1` mit 2 bis 4 Envs und **synthetischen** Clips `[T, 591]` (stehender Mensch, ruhendes Objekt): Aufbau, Reset, 60 Policy-Schritte mit Null-Aktion, einzelne Physikschritte.
- Ein Minimaltest, was nach `simulation_app.close()` noch läuft.

Nicht ausgeführt wurden: ein Training, der Export, Teacher und Student als Policy sowie alles mit echten Daten (`InterAct/` liegt lokal nicht vor). Wo eine Aussage aus einem Lauf stammt, steht «gemessen». Aussagen über Isaac Lab selbst (Kapitel 5) stützen sich auf den Quelltext des Forks unter `~/Documents/repos/IsaacLab-MRL`.

Schwerpunkt: alles bis und mit Retargeting sowie alles, was Isaac Lab benutzt. Die MuJoCo-Skripte (`sim2sim_*.py`, `utils/obs*.py`) und die Netzarchitektur des Students werden nur gestreift.

## Inhalt

1. [Das Wichtigste vorweg](#1-das-wichtigste-vorweg)
2. [Aufbau des Repos](#2-aufbau-des-repos)
3. [Die Pipeline in fünf Stufen](#3-die-pipeline-in-fünf-stufen)
4. [Eingabedaten: `[T, 591]`](#4-eingabedaten-t-591)
5. [Wie Isaac Lab funktioniert](#5-wie-isaac-lab-funktioniert)
6. [Wie das Repo Isaac Lab benutzt](#6-wie-das-repo-isaac-lab-benutzt)
7. [Stufe 1: Retargeting-Policy trainieren](#7-stufe-1-retargeting-policy-trainieren)
8. [Stufe 2: Export nach `[T, 630]`](#8-stufe-2-export-nach-t-630)
9. [Teacher und Student in Isaac Lab](#9-teacher-und-student-in-isaac-lab)
10. [Auffälligkeiten und Stolpersteine](#10-auffälligkeiten-und-stolpersteine)

---

## 1. Das Wichtigste vorweg

- **Es gibt keine SMPL-X-Verarbeitung im Repo.** Die Pipeline startet bei fertig aufbereiteten Tensoren `[T, 591]` im InterMimic-Format. Das SMPL-X-Körpermodell wird nirgends geladen; die Umrechnung von SMPL-X-Parametern in diese Tensoren passiert ausserhalb (InterMimic / InterAct).
- **Das Retargeting ist kein IK-Verfahren, sondern eine RL-Policy in der Simulation.** Der G1 lernt per PPO, die auf 0.8 skalierten menschlichen Keypoints samt Objekt physikalisch nachzufahren. Der aufgezeichnete Simulationszustand dieses Rollouts ist der retargetete Datensatz `[T, 630]`.
- **Die Portierung hat die Simulation ausgetauscht, nicht die Aufgaben.** Alles, was Isaac Lab berührt, liegt in fünf Dateien unter `ultra/isaac/` plus `scripts/convert_assets.py`. Die Task-Klassen (Observation, Reward, Reset) rechnen weiter auf denselben Tensoren wie in der Isaac-Gym-Version: Quaternionen als (x, y, z, w), Körper und Gelenke in der Isaac-Gym-Reihenfolge, Positionen relativ zum Env-Ursprung. `UltraBaseEnv` übersetzt bei jedem Lesen und Schreiben zwischen diesem «Legacy-Layout» und Isaac Lab.
- **Deshalb bleiben Datensätze, Checkpoints, MuJoCo-Sim2sim und Deployment gültig.** Geändert hat sich die Physik darunter (PhysX 5 aus Isaac Sim statt der PhysX-Version von Isaac Gym), sodass alte Policies nicht zwingend gleich gut laufen.
- **Von Isaac Labs `DirectRLEnv` wird nur der Aufbau benutzt.** Den Schrittablauf (`step`, Reset, Dones) macht das Repo selbst, weil der rl-games-Agent die Envs von aussen zurücksetzt.

---

## 2. Aufbau des Repos

```
scripts/                     Einstiegspunkte pro Stufe (Shell + Python)
  convert_assets.py          URDF/OBJ → USD (einmalig, nötig für Isaac Lab)
  check_layout.py            Prüft die Legacy-Zuordnung gegen MuJoCo-Vorwärtskinematik
  prepare_retarget_smplx.py  Clips filtern, Symlinks mit Asset-Skalierung
  export_retarget_smplx.py   Stufe 2: Rollouts als [T, 630] speichern
  convert_rlg_checkpoint.py  rl-games-1.1.4-Checkpoint ins 1.6-Format umschreiben
ultra/run.py                 Einstieg für alle Stufen (AppLauncher + rl-games-Runner)
ultra/run_distill.py         nur noch ein Alias auf run.py
ultra/run_teacher_inference.py   Teacher auf einem Clip ausrollen und speichern
ultra/isaac/                 die Isaac-Lab-Schicht
  scene_cfg.py               YAML → UltraEnvCfg: Sim, Roboter, Objekt, Sensoren, Boden
  base_env.py                UltraBaseEnv(DirectRLEnv): Legacy-Tensoren, Lesen/Schreiben, Physikschritt
  legacy_layout.py           Körper-/Gelenkreihenfolge, PD-Gains, Index- und Quaternion-Umrechnung
  vec_task.py                Wrapper zwischen Task und rl-games
  viewer.py                  Kamera, Debug-Zeichnen, Bildaufnahme
ultra/utils/config.py        CLI-Argumente, YAML laden
ultra/utils/parse_task.py    Task bei gymnasium registrieren und erzeugen
ultra/utils/gym_torch_utils.py   Nachbau von isaacgym.torch_utils (Quaternion-Mathematik, xyzw)
ultra/utils/torch_utils.py   weitere Quaternion-Hilfen (Heading, exp-map, 6D)
ultra/env/tasks/             alle Umgebungen (Observation, Reward, Reset)
ultra/learning/              PPO-Agent, Netze, Player (Unterklassen von rl-games 1.6.1)
ultra/data/cfg/              Env-YAMLs; train/rlg/ = PPO-YAMLs
ultra/data/assets/g1/        G1 mit 29 Freiheitsgraden als URDF und als XML (MuJoCo)
ultra/data/assets/objects/   Objekt-URDFs/-Meshes, Skalierung 080 und 100
ultra/data/assets/usd/       konvertierte USD-Dateien (erzeugt, nicht in git)
ultra/weights/               mitgelieferter Teacher-Checkpoint
ultra/sim2sim_*.py, utils/obs*.py, export_jit.py   MuJoCo und Deployment, ohne Isaac Lab
```

Gegenüber der Isaac-Gym-Version sind weggefallen: `env/tasks/base_task.py`, `vec_task.py`, `vec_task_wrappers.py` (ersetzt durch `ultra/isaac/`) sowie die kopierten rl-games-Dateien `learning/a2c_common.py`, `central_value.py`, `models.py`, `network_builder.py` (ersetzt durch das installierte rl-games 1.6.1).

### Klassenhierarchie der Tasks

| Klasse | Datei | Rolle |
|---|---|---|
| `DirectRLEnv` | Isaac Lab | Simulation und Szene anlegen |
| `UltraBaseEnv` | `isaac/base_env.py` | Legacy-Tensoren, Lesen/Schreiben der Simulation, `step()`-Gerüst |
| `Humanoid_SMPLX` | `env/tasks/humanoid.py` | Views auf die Tensoren, Physikschleife, Reset, Basis-Observations |
| `Humanoid_G1` | `env/tasks/humanoid_g1.py` | G1-Eigenschaften, PD-Gains, Reward und Observations für Stufe 1 |
| `Ultra` | `env/tasks/ultra.py` | Motion-Dateien, Objektpunkte, Reset aus der Referenz |
| `UltraG1` | `env/tasks/ultra_g1.py` | **Retargeting-Task** (Stufe 1 und 2) |
| `UltraG1Retarget` | `env/tasks/ultra_g1_retarget.py` | trotz des Namens der **Teacher** (Stufe 3) |
| `UltraDistillObjV2Point` | `env/tasks/ultra_g1_distill_obj_v2vae.py` | Student-Distillation (Stufe 4) |
| `UltraDistillObjV3RL` | `env/tasks/ultra_g1_distill_obj_v3rl.py` | Student-Finetuning mit RL (Stufe 5) |

`UltraG1(Humanoid_G1, Ultra)` nutzt Mehrfachvererbung. Die Auflösungsreihenfolge (gemessen):

```
UltraG1 → Humanoid_G1 → Ultra → Humanoid_SMPLX → UltraBaseEnv → DirectRLEnv
```

Jeder Task hat zwei Namen: den alten Klassennamen und eine gymnasium-ID (`ultra/utils/parse_task.py:36`). `--task` nimmt beide.

| Klassenname | gymnasium-ID |
|---|---|
| `UltraG1` | `Ultra-G1-RetargetSMPLX-v0` |
| `UltraG1Retarget` | `Ultra-G1-Teacher-v0` |
| `UltraDistillObjV2Point` | `Ultra-G1-Student-v0` |
| `UltraDistillObjV3RL` | `Ultra-G1-Finetune-v0` |

### Aufrufkette beim Start

```
scripts/train_retarget_smplx.sh
  ├─ python scripts/prepare_retarget_smplx.py     Symlinks der unterstützten Clips
  └─ python ultra/run.py --task UltraG1 --cfg_env … --cfg_train … --headless
       ├─ get_args_parser() + AppLauncher.add_app_launcher_args()
       ├─ AppLauncher(args)      startet Isaac Sim; erst danach weitere Imports
       ├─ load_cfg()             beide YAMLs laden, CLI-Overrides anwenden
       └─ rl-games Runner        Agent "ultra" = UltraAgent (PPO)
            └─ RLGPUEnv → create_rlgpu_env()
                 └─ parse_task()
                      ├─ make_env_cfg(cfg)          YAML-Dict → UltraEnvCfg (mit SimulationCfg)
                      ├─ gym.make(id, cfg=env_cfg)  → UltraG1(cfg)
                      └─ UltraVecTask(task)         Wrapper für rl-games
```

`run.py` registriert die Umgebung unter dem Namen `rlgpu` bei rl-games und die eigenen Klassen für Agent, Player, Modell und Netz unter `ultra`. `python ultra/run.py` setzt `ultra/` an den Anfang von `sys.path`; deshalb lauten die Imports `from isaac…`, `from env.tasks…`, `from utils…`.

---

## 3. Die Pipeline in fünf Stufen

| Stufe | Eingabe → Ausgabe | Einstieg | Task |
|---|---|---|---|
| 0 Assets | URDF/OBJ → USD | `scripts/convert_assets.py` | – |
| 1 Retargeting | `[T, 591]` Mensch → Policy | `scripts/train_retarget_smplx.sh` | `UltraG1` |
| 2 Export | `[T, 591]` + Policy → `[T, 630]` G1 | `scripts/export_retarget_smplx.py` | `UltraG1` |
| 3 Teacher | `[T, 630]` → Tracking-Policy | `scripts/train_teacher.sh` | `UltraG1Retarget` |
| 4 Student | `[T, 630]` + Teacher → Student | `scripts/train_student.sh` | `UltraDistillObjV2Point` |
| 5 Finetuning | Student → zielgerichtete Policy | `scripts/train_finetune.sh` | `UltraDistillObjV3RL` |

Stufe 0 ist neu und einmalig nötig. Die Kommentare in den Shell-Skripten zählen anders als das README (`train_teacher.sh` nennt sich dort «Stage 2»). Diese Erklärung folgt der Zählung des README.

Auf einer GPU mit 16 GB passen die voreingestellten 4096 Envs nicht; `docs/install.md` nennt `--num_envs 2048`. `horizon_length · num_envs` muss durch `minibatch_size` teilbar bleiben (Stufe 1: 32 · 2048 / 16384 = 4).

---

## 4. Eingabedaten: `[T, 591]`

Eine Datei ist ein Tensor mit einer Zeile pro Frame bei 30 fps. So liest `UltraG1._load_motion` (`ultra/env/tasks/ultra_g1.py:75`) die Spalten:

| Spalten | Inhalt |
|---|---|
| 0:3 | Root-Position |
| 3:7 | Root-Quaternion (xyzw) |
| 9:162 | 153 Gelenkwerte (51×3); für den G1 nicht im Reward verwendet |
| 162:318 | Positionen der 52 Körper |
| 318:321 | Objekt-Position |
| 321:325 | Objekt-Quaternion |
| 330 | Kontakt-Label des Objekts |
| 331:383 | Kontakt-Label pro Körper (52) |
| 383:591 | Rotationen der 52 Körper (Quaternionen) |

Die Spalten 7:9 und 325:330 liest der Loader nicht.

Der Dateiname trägt Information: `sub10_largebox_003_080_080_080.pt`. `Ultra.__init__` liest daraus den Objektnamen (zweites Feld) und die Asset-Skalierung (die letzten drei Felder) und bildet daraus den Asset-Namen `largebox_080_080_080`. Aus der sortierten Menge dieser Namen entsteht die Objektliste der Szene.

`scripts/prepare_retarget_smplx.py` rechnet nichts um. Es filtert die Rohdateien (`subNN_objekt_NNN.pt`) auf die vier Objekte mit vorhandenen Assets (`largebox`, `plasticbox`, `smallbox`, `suitcase`) und legt Symlinks mit angehängter Asset-Skalierung an.

Die Daten sind unabhängig vom Simulator. An diesem Kapitel hat die Portierung nichts geändert.

---

## 5. Wie Isaac Lab funktioniert

Isaac Lab ist NVIDIAs Nachfolger von Isaac Gym. Die Grundidee ist dieselbe: Tausende Kopien einer Szene laufen in einer einzigen PhysX-Simulation auf der GPU, und der Zustand liegt als PyTorch-Tensor auf derselben GPU wie das Netz. Anders ist der Unterbau: Isaac Lab ist kein eigenständiger Simulator, sondern eine Python-Bibliothek auf Isaac Sim, und die Szene ist eine USD-Datei statt einer Liste von API-Aufrufen.

### 5.1 Die Schichten

| Schicht | Was sie ist |
|---|---|
| Omniverse Kit | Anwendungsrahmen: Erweiterungen, Fenster, Renderer, Ereignisschleife |
| USD (`pxr`) | Szenenbeschreibung: ein Baum aus «Prims» mit Attributen und Schemas |
| PhysX 5 | Physik; liest die Physik-Schemas aus der USD-Szene |
| Isaac Sim 5.1 | Kit-Anwendung mit Robotik-Erweiterungen, URDF-Import, Tensor-Schnittstelle zu PhysX |
| Isaac Lab 2.3.2 | Python-Bibliothek darüber: Assets, Szene, Sensoren, Aktuatoren, Env-Klassen |

Im Repo kommt davon vor: `isaaclab.app`, `isaaclab.sim`, `isaaclab.envs`, `isaaclab.assets`, `isaaclab.actuators`, `isaaclab.scene`, `isaaclab.sensors`, `isaaclab.terrains`, `isaaclab.sim.converters`, dazu `pxr` direkt im Konvertierungsskript. Das sind alles Module, die es auch im offiziellen Isaac Lab gibt; Aufrufe, die nur der Fork hätte, habe ich im Code nicht gefunden.

### 5.2 Start: erst die App, dann der Rest

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)        # startet Kit / Isaac Sim
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils        # erst jetzt möglich
```

Die meisten `isaaclab.*`-Module und `pxr` lassen sich erst importieren, wenn die App läuft. Deshalb stehen in `run.py`, `convert_assets.py` und `check_layout.py` Imports mitten in der Datei (mit `# noqa: E402`). Das ersetzt die alte Regel «`isaacgym` vor `torch` importieren».

`AppLauncher.add_app_launcher_args(parser)` ergänzt den eigenen Parser um `--headless`, `--device`, `--enable_cameras` und weitere. Am Ende steht `simulation_app.close()`. Gemessen: Dieser Aufruf beendet den Prozess selbst; Code dahinter läuft nicht mehr (siehe Kapitel 10).

### 5.3 Szene: USD und Klonen

Eine Isaac-Lab-Szene ist ein USD-Baum:

```
/World
├─ ground                    Bodenebene, gemeinsam für alle Envs
├─ Light
└─ envs
   ├─ env_0
   │  ├─ Robot               Artikulation (Links als Prims, Gelenke dazwischen)
   │  └─ Object              Starrkörper
   ├─ env_1
   │  └─ …
```

Man beschreibt die Szene mit Konfigurationsklassen (`@configclass`, eine Dataclass-Variante): ein `InteractiveSceneCfg` mit einem Feld pro Asset oder Sensor. Der Prim-Pfad enthält einen regulären Ausdruck, z. B. `/World/envs/env_.*/Robot`. `InteractiveScene` baut `env_0` und klont es `num_envs`-mal. Die Envs liegen auf einem Gitter; `scene.env_origins` gibt den Ursprung jedes Env in Weltkoordinaten.

Zwei Schalter sind hier wichtig:

- `replicate_physics=True` lässt PhysX ein Env parsen und den Rest kopieren. Das geht nur, wenn alle Envs identisch sind. Mit unterschiedlichen Objekten pro Env muss es `False` sein.
- `filter_collisions` (Standard an) sorgt dafür, dass Envs einander nicht sehen; global angelegte Prims wie der Boden kollidieren mit allen. Das entspricht der Kollisionsgruppe pro Env aus Isaac Gym.

Assets kommen als USD-Dateien. URDF und Meshes muss man vorher konvertieren (`UrdfConverter`, `MeshConverter`). Dabei werden Physik-Eigenschaften als USD-Schemas in die Datei geschrieben (Masse, Kollisionsform, konvexe Zerlegung, Kontaktabstände). Beim Spawnen lassen sich viele davon über `spawn=…Cfg(...)` überschreiben.

Wie in Isaac Gym steht die Szene nach dem Start der Simulation fest. Welches Objekt in welchem Env liegt, wird beim Aufbau entschieden.

### 5.4 Assets und ihre Datenpuffer

Statt weniger flacher Tensoren über alle Actors hat jedes Asset ein eigenes Objekt mit eigenen Tensoren:

| Klasse | Für | Wichtige Felder von `.data` |
|---|---|---|
| `Articulation` | Roboter | `root_state_w` (N, 13), `joint_pos` / `joint_vel` (N, J), `body_state_w` (N, B, 13), `applied_torque` (N, J), `joint_pos_limits` (N, J, 2) |
| `RigidObject` | Objekt | `root_state_w` (N, 13) |

Lesen: Die Felder von `.data` werden bei Bedarf aus der Simulation geholt und pro Zeitschritt zwischengespeichert. `scene.update(dt)` setzt den Zeitstempel weiter; ohne diesen Aufruf bekäme man alte Werte. Das entspricht `refresh_*_tensor`.

Schreiben gibt es in zwei Arten:

- **Zustand setzen** (Reset): `write_root_state_to_sim(state, env_ids)` und `write_joint_state_to_sim(pos, vel, env_ids=…)`. Das wirkt sofort und ersetzt `set_*_tensor_indexed`. Indiziert wird mit Env-Indizes, nicht mit globalen Actor-Indizes.
- **Stellgrössen setzen**: `set_joint_position_target(...)` oder `set_joint_effort_target(...)` schreiben nur in einen Puffer. Erst `scene.write_data_to_sim()` gibt sie an PhysX weiter.

Für alles, was Isaac Lab nicht kapselt, gibt es `asset.root_physx_view`, die rohe Tensor-Schnittstelle von PhysX: `get_masses` / `set_masses`, `set_coms`, `set_inertias`, `get_material_properties` / `set_material_properties`. Das Repo ruft sie mit CPU-Tensoren und Env-Indizes auf, so wie Isaac Labs eigene Randomisierungsfunktionen.

Ein Unterschied zu Isaac Gym, der für Resets zählt: Beim Lesen der Link-Posen ruft Isaac Lab erst `update_articulations_kinematic()` auf. Die Link-Posen folgen deshalb schon direkt nach dem Setzen von Root- und Gelenkzustand der neuen Pose, ohne Physikschritt. Gemessen: Der linke Knöchel liegt vor dem ersten Reset auf 0.133 m und unmittelbar danach, ohne Physikschritt, auf einem neuen Wert (0.201 m in der Gegenprobe aus Kapitel 10). Der Hinweis der alten Erklärung, nach einem Reset zeige der Körper-Tensor noch den alten Zustand, gilt hier nicht mehr.

### 5.5 Aktuatoren

Gelenkantriebe sind in Isaac Lab eigene Objekte, die zwischen Stellgrösse und Simulation sitzen.

| Modell | Wer rechnet das Moment |
|---|---|
| `ImplicitActuator` | PhysX: `stiffness · (Ziel − q) − damping · q̇`, implizit im Solver |
| `IdealPDActuator` und andere explizite | Isaac Lab in Python, einmal pro Physikschritt; an PhysX geht ein Moment |

Beim impliziten Modell setzt Isaac Lab `stiffness`, `damping`, `armature` und `effort_limit_sim` als Gelenkeigenschaften in PhysX. Mit `stiffness = damping = 0` bleibt ein reiner Drehmoment-Eingang übrig. Das deckt beide Antriebsmodi aus Isaac Gym ab (`DOF_MODE_POS` und `DOF_MODE_EFFORT`).

PhysX gibt das Moment des impliziten Reglers nicht heraus. `ImplicitActuator.compute` rechnet deshalb beim Schreiben eine Näherung aus dem Gelenkzustand vor dem Schritt und legt sie, auf das Effort-Limit begrenzt, in `data.applied_torque` ab.

### 5.6 Sensoren

Kontaktkräfte kommen nicht aus einem globalen Tensor, sondern aus einem `ContactSensor`. Er braucht zwei Dinge: Die Körper müssen mit `activate_contact_sensors=True` gespawnt sein, und der Sensor bekommt einen Prim-Pfad-Ausdruck für die Körper, die er meldet. `sensor.data.net_forces_w` hat die Form (N, Körper, 3) und enthält die Summe aller Kontaktkräfte auf jeden Körper, gemittelt über den letzten Physikschritt. `update_period=0.0` heisst: bei jedem Physikschritt neu.

Die Reihenfolge der Körper im Sensor muss nicht die der Artikulation sein (gemessen: sie ist es beim G1 nicht).

### 5.7 `DirectRLEnv`

Isaac Lab kennt zwei Arten, eine Umgebung zu schreiben: «manager-based» (Observation, Reward, Events als konfigurierte Bausteine) und «direct» (eine Klasse, die alles selbst rechnet). `DirectRLEnv` ist die zweite und liegt nahe am Stil von Isaac Gym.

Der Konstruktor macht (`isaaclab/envs/direct_rl_env.py`):

1. `SimulationContext(cfg.sim)` anlegen. Es darf nur einen geben.
2. `InteractiveScene(cfg.scene)` bauen und `_setup_scene()` der Unterklasse rufen.
3. `sim.reset()`: startet die Physik. Erst danach existieren die Tensor-Views. Das entspricht `prepare_sim`.
4. `scene.update(dt)`, Puffer und gymnasium-Spaces anlegen.

Vorgesehen ist dann, dass die Unterklasse `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones` und `_reset_idx` füllt und `DirectRLEnv.step()` den Ablauf steuert, inklusive automatischem Reset fertiger Envs. Das Repo benutzt diesen Teil nicht (siehe 6.7).

### 5.8 Simulationsschritt und Zeit

- `cfg.sim.dt` ist der Physikschritt.
- `cfg.decimation` ist die Zahl der Physikschritte pro Policy-Schritt (`controlFrequencyInv` in Isaac Gym).
- Ein Physikschritt von Hand sind drei Aufrufe: `scene.write_data_to_sim()`, `sim.step(render=False)`, `scene.update(dt)`.
- Gerendert wird getrennt mit `sim.render()`.

### 5.9 Konventionen, die von Isaac Gym abweichen

| | Isaac Gym | Isaac Lab |
|---|---|---|
| Quaternion | (x, y, z, w) | (w, x, y, z) |
| Positionen | relativ zum Env | Weltkoordinaten; Env-Ursprung in `scene.env_origins` |
| Reihenfolge Gelenke/Körper | Tiefensuche, Geschwister nach Namen | Breitensuche |
| Indizes beim Schreiben | globale Actor-Indizes | Env-Indizes pro Asset |
| Zustands-Tensoren | Views auf Simulator-Speicher | Kopien in `.data` |

Gemessene Gelenkreihenfolge von Isaac Lab beim G1: `left_hip_pitch`, `right_hip_pitch`, `waist_yaw`, `left_hip_roll`, `right_hip_roll`, `waist_roll`, … Die Isaac-Gym-Reihenfolge ist dagegen linkes Bein, rechtes Bein, Hüfte/Taille, linker Arm, rechter Arm.

### 5.10 Von Isaac Gym nach Isaac Lab

| Isaac Gym | Isaac Lab |
|---|---|
| `acquire_gym`, `create_sim` | `AppLauncher`, `SimulationContext(SimulationCfg)` |
| `load_asset(URDF)` | USD-Datei + `ArticulationCfg` / `RigidObjectCfg` |
| `create_env`, `create_actor` in einer Schleife | `InteractiveSceneCfg`, Klonen |
| `prepare_sim` | `sim.reset()` |
| `acquire_*_tensor` + `wrap_tensor` | `asset.data.*` |
| `refresh_*_tensor` | `scene.update(dt)` |
| `set_actor_root_state_tensor_indexed` | `asset.write_root_state_to_sim(state, env_ids)` |
| `set_dof_state_tensor_indexed` | `robot.write_joint_state_to_sim(pos, vel, env_ids=…)` |
| `set_dof_position_target_tensor` | `robot.set_joint_position_target` + `write_data_to_sim` |
| `set_dof_actuation_force_tensor` | `robot.set_joint_effort_target` + `write_data_to_sim` |
| `simulate` + `fetch_results` | `sim.step(render=False)` |
| Net-Contact-Force-Tensor | `ContactSensor.data.net_forces_w` |
| DOF-Force-Tensor | `robot.data.applied_torque` |
| DOF-Properties (`driveMode`, `stiffness`, …) | `ImplicitActuatorCfg` |
| Kollisionsgruppe pro Env | `filter_collisions` der Szene |
| Kollisionsfilter-Bitmaske | `UsdPhysics.FilteredPairsAPI` in der USD-Datei |
| VHACD beim Laden | konvexe Zerlegung als Schema in der USD-Datei |
| Shape-/Body-Properties pro Actor | `root_physx_view.set_material_properties`, `set_masses`, … |
| `set_sim_params` (Schwerkraft) | `sim.physics_sim_view.set_gravity` |
| `create_viewer`, `draw_viewer` | Kit-Viewport, `sim.render()` |
| `gymutil.draw_lines` | Erweiterung `isaacsim.util.debug_draw` |

---

## 6. Wie das Repo Isaac Lab benutzt

### 6.1 Die Grundidee: Legacy-Layout

Die Task-Klassen wurden gegen die Tensoren von Isaac Gym geschrieben. Statt sie umzuschreiben, stellt `UltraBaseEnv` dieselben Tensoren in derselben Form bereit (`ultra/isaac/base_env.py:127`):

| Tensor | Form | Inhalt |
|---|---|---|
| `_root_states` | (2·N, 13) | pro Env Roboter-Root, dann Objekt-Root; Quaternion xyzw, Position relativ zum Env |
| `_dof_state` | (29·N, 2) | Gelenkposition und -geschwindigkeit in `LEGACY_DOF_NAMES`-Reihenfolge |
| `_rigid_body_state` | (40·N, 13) | die 39 G1-Körper in `LEGACY_BODY_NAMES`-Reihenfolge, dann das Objekt |
| `_contact_force_state` | (40·N, 3) | Kontaktkraft pro Körper, gleiche Anordnung |
| `dof_force_tensor` | (N, 29) | Gelenkmomente |

Das sind normale PyTorch-Tensoren, keine Views auf Simulator-Speicher. Zwei Methoden halten sie mit der Simulation in Einklang:

- `_refresh_sim_tensors()` (`base_env.py:164`) kopiert aus `robot.data`, `object.data` und den Kontaktsensoren in die Legacy-Tensoren: Env-Ursprung abziehen, Quaternion von wxyz nach xyzw rollen, Gelenke und Körper permutieren. Es schreibt in die bestehenden Tensoren hinein, damit die Views der Task-Klassen gültig bleiben.
- `_set_actor_root_state_indexed(actor_ids)` und `_set_dof_state_indexed(actor_ids)` gehen den umgekehrten Weg. Sie nehmen weiterhin die alten globalen Actor-Indizes (`2·env` für den Roboter, `2·env + 1` für das Objekt), rechnen sie in Env-Indizes um und rufen `write_root_state_to_sim` bzw. `write_joint_state_to_sim`.

Die Permutation steht in `ultra/isaac/legacy_layout.py`. `LEGACY_BODY_NAMES` (39) und `LEGACY_DOF_NAMES` (29) sind dort ausgeschrieben; `legacy_to_sim_index` baut daraus einen Index-Tensor `idx` mit `sim_tensor[..., idx] == legacy_tensor`. Es gibt drei solche Tensoren (`base_env.py:114`): für Gelenke, für Körper und für den Kontaktsensor, dessen Reihenfolge eine eigene ist. Stimmen Körper- oder Gelenkzahl der USD nicht (39 / 29), bricht der Aufbau mit dem Hinweis auf `convert_assets.py --force` ab.

`legacy_layout.py` importiert nichts von Isaac Lab und wird auch von den MuJoCo-Werkzeugen benutzt. Dort stehen auch die Antriebsparameter pro Gelenk (`G1_STIFFNESS`, `G1_DAMPING`, `G1_ARMATURE`, `G1_EFFORT`), die früher fest in `Humanoid_G1._build_env` standen.

### 6.2 Assets konvertieren

`scripts/convert_assets.py` schreibt nach `ultra/data/assets/usd/` (git-ignoriert).

**G1** (`convert_g1`, Zeile 127): `UrdfConverter` mit

| Einstellung | Wert | Grund |
|---|---|---|
| `merge_fixed_joints` | `False` | Alle 39 Links bleiben eigene Körper. Datensätze und Policies indizieren sie. |
| `fix_base` | `False` | frei bewegliche Basis |
| `collider_type` | `convex_decomposition` | wie VHACD in Isaac Gym |
| `self_collision` | `True` | Selbstkollision an |
| `joint_drive` | `None` | Antriebe kommen erst aus der `ActuatorCfg` |

Danach bearbeitet `postprocess_g1` (Zeile 83) die USD direkt mit `pxr`:

- Links ohne `<inertial>` im URDF (Sensor- und Hilfsframes) bekommen Masse 1 g und Trägheit 1e-6.
- Alle Kollisionsformen bekommen `contact_offset = 0.02` und `rest_offset = 0.0`; Mesh-Kollider die Zerlegungsparameter (5 Hüllen, 16 Eckpunkte pro Hülle, Auflösung 60000). Das muss in die USD, weil der Importer die Kollider als instanzierte Prototypen ablegt, die sich beim Spawnen nicht mehr ändern lassen.
- Die Kollisionsfilter-Bits aus Isaac Gym (rechts Knöchel 2, Knie 6, Hüfte 12; links 16, 48, 96) werden zu expliziten Paaren: Für je zwei Bein-Links, deren Bits sich überschneiden, wird ein Eintrag in `FilteredPairsAPI` geschrieben. Aus den Link-Namen ergeben sich 9 Paare pro Bein: die Hüft-Links untereinander, die Knöchel-Links untereinander, Knöchel gegen Knie und Knie gegen Hüfte. Knöchel gegen Hüfte kollidiert weiterhin, ebenso linkes gegen rechtes Bein und Arme gegen Rumpf.

**Objekte** (`convert_objects`, Zeile 146): `MeshConverter` pro OBJ unter `objects/diverse/`, je zweimal: mit 5 Hüllen (`…_h5.usd`, für das Retargeting) und mit 10 Hüllen (`…_h10.usd`, sonst). Dichte und Skalierung werden beim Spawnen gesetzt.

Die 8 mitgelieferten Objekte ergeben 16 USD-Dateien. Nach Änderungen an URDF oder Meshes: `--force`.

### 6.3 Konfiguration und Simulationsparameter

Die Task-Parameter stehen weiter in den YAML-Dateien. `make_env_cfg` (`ultra/isaac/scene_cfg.py:178`) verpackt das YAML-Dict in ein `UltraEnvCfg`:

- `cfg.task` ist das YAML-Dict. `UltraEnvCfg` leitet `cfg["env"]`, `cfg.get(...)` und `in` dorthin weiter. So funktioniert `self.cfg["env"]["scaling"]` im Task-Code unverändert, obwohl `self.cfg` jetzt ein Isaac-Lab-Config-Objekt ist.
- `cfg.sim`, `cfg.decimation`, `cfg.seed` sind die Isaac-Lab-Felder.
- `episode_length_s = 1e6`: Isaac Labs eigenes Zeitlimit ist abgeschaltet.

Woher die Simulationsparameter kommen (`build_sim_cfg`, `build_robot_cfg`, `build_object_cfg`):

| Parameter | Wert | Herkunft | Wo gesetzt |
|---|---|---|---|
| `dt` | 1 ms | `SIM_DT` in `scene_cfg.py` | `SimulationCfg` |
| `decimation` | 17 | YAML `env.controlFrequencyInv` | `UltraEnvCfg` |
| Solver | TGS | YAML `sim.physx.solver_type` | `PhysxCfg` |
| `bounce_threshold_velocity` | 0.2 | YAML | `PhysxCfg` |
| Positions-/Geschwindigkeits-Iterationen | 4 / 1 | YAML | pro Artikulation und pro Objekt |
| `max_depenetration_velocity` | 1.0 | YAML | pro Körper |
| `contact_offset` / `rest_offset` | 0.02 / 0.0 | G1: fest im Konvertierungsskript; Objekt: YAML | USD bzw. Spawn |
| Winkeldämpfung der Körper | 0.5 | Konstante (Isaac-Gym-Standard) | `RigidBodyPropertiesCfg` |
| max. Winkel-/Lineargeschwindigkeit | 64 / 1000 | Konstante | `RigidBodyPropertiesCfg` |
| Boden: Reibung, Rückprall | 1.0 / 0 | YAML `env.plane` | `TerrainImporterCfg` |
| Hochachse, Schwerkraft | z, −9.81 | Isaac-Lab-Standard | – |

Iterationen sind in PhysX 5 eine Eigenschaft der Artikulation bzw. des Körpers, nicht mehr der Szene. `sim.substeps`, `num_threads`, der `flex`-Block und die GPU-Puffergrössen aus Isaac Gym werden nicht gelesen.

Ein Policy-Schritt sind 17 Physikschritte zu 1 ms, zusammen 17 ms oder etwa 58.8 Hz (gemessen: `dt = 0.017`). Die Referenzdaten werden als 60 Hz behandelt und pro Policy-Schritt um einen Frame weitergezählt. Die Abweichung von rund 2 % wird hingenommen.

### 6.4 Szenenaufbau und Reihenfolge im Konstruktor

Die Szene hängt von den Daten ab: Welche Objekte es gibt, steht in den Dateinamen der Motions. Deshalb wird die Konfiguration in zwei Schritten fertig, und die Reihenfolge im Konstruktor ist wichtig:

```
UltraG1.__init__
└─ Humanoid_G1.__init__            keyIndex/contactIndex als Tensoren
   └─ Ultra.__init__               Motion-Dateien auflisten → Objektnamen, object_id pro Motion
      └─ Humanoid_SMPLX.__init__   YAML lesen, Observation- und Aktionsgrösse bestimmen
         └─ UltraBaseEnv.__init__
            ├─ finalize_env_cfg()              Szene in die Config eintragen (braucht Objektnamen, numObs)
            ├─ DirectRLEnv.__init__            Sim anlegen, Szene bauen und klonen, sim.reset()
            ├─ _build_index_maps()             Legacy ↔ Isaac Lab
            ├─ _allocate_state_tensors()       die Tensoren aus 6.1
            ├─ _allocate_task_buffers()        obs_buf, rew_buf, reset_buf, progress_buf, …
            ├─ _check_object_assignment()      Env i hat Objekt i % Anzahl
            ├─ UltraViewer                     nur ohne --headless oder mit --save_images
            ├─ _setup_env_properties()         Objektpunkte, Gelenkgrenzen, Gains, Reibung, DR
            └─ _refresh_sim_tensors()
         (zurück in Humanoid_SMPLX)            Views auf die Tensoren, Body-Indizes
      (zurück in Ultra)                        _load_motion(), Objekt-Views
   (zurück in Humanoid_G1)                     Puffer für Aktionen, SmoothRewards
(zurück in UltraG1)                            scaling, initRootHeight, init_dof
```

`_setup_env_properties` ist der Haken für alles, was in Isaac Gym beim Anlegen der Actors passierte und jetzt erst nach dem Start der Simulation möglich ist. Für `UltraG1` läuft dort: `_load_target_asset` (Objektpunkte), `Humanoid_G1._setup_env_properties` (Fuss-, Knie-, Torso-Indizes, Gelenkgrenzen, Gains, Reibung, Masse) und `_setup_target_properties` (Objektmaterial).

Die Szene selbst (`finalize_env_cfg`, `scene_cfg.py:194`):

| Feld | Prim-Pfad | Inhalt |
|---|---|---|
| `robot` | `/World/envs/env_.*/Robot` | `ArticulationCfg` mit der G1-USD, ein `ImplicitActuatorCfg` für alle Gelenke |
| `object` | `/World/envs/env_.*/Object` | `RigidObjectCfg` mit `MultiAssetSpawnerCfg` über die Objekt-USDs |
| `robot_contact` | `/World/envs/env_.*/Robot/.*` | Kontaktsensor auf allen 39 Links |
| `object_contact` | `/World/envs/env_.*/Object` | Kontaktsensor auf dem Objekt |
| `terrain` | `/World/ground` | Ebene |

Dazu `num_envs`, `env_spacing` (1 m) und `replicate_physics=False`, weil die Envs unterschiedliche Objekte haben. Das Licht kommt aus `_setup_scene`.

Einzelheiten:

- **Objektwahl.** `MultiAssetSpawnerCfg(random_choice=False)` gibt Env i das Objekt `i % Anzahl Objekttypen`, wie in Isaac Gym. `_check_object_assignment` prüft, dass die Reihenfolge der Objekt-Prims der Env-Reihenfolge entspricht. Gemessen mit zwei Objekttypen und 4 Envs: `[0, 1, 0, 1]`.
- **Objektmasse.** Dichte 25 (`objectDensity`) beim Spawnen; die Masse ergibt sich aus dem Volumen. Gemessen: `largebox_080` 0.49 kg, `suitcase_080` 0.57 kg. Skalierung über `ballSize`.
- **Hüllenzahl.** 5 Hüllen im Retargeting, sonst 10.
- **Startposen beim Spawnen.** Roboter auf (0, 0, 0.89) mit Gelenken auf 0, Objekt auf (1, 0, 0.5), damit vor dem ersten Reset nichts ineinander steckt.
- **Env-Abstand.** Mit 1 m Abstand überlappen sich die Envs räumlich, sobald sich ein Roboter bewegt. Das ist nur optisch; sie kollidieren nicht miteinander. Gemessene Ursprünge für 4 Envs: (±0.5, ±0.5, 0).
- **Objektpunkte.** `Ultra._load_target_asset` (`ultra/env/tasks/ultra.py:232`) lädt weiterhin das OBJ mit `trimesh` und sampelt 256 Oberflächenpunkte relativ zum Schwerpunkt der Vertices. Diese Punkte (`self.object_points`) sind die Grundlage für alle Hand-Objekt-Distanzen in Reward und Observation; sie existieren nur in PyTorch, nicht in der Simulation. Das simulierte Objekt kommt aus der USD.

Der G1 hat 39 Körper und 29 Gelenke, gemessene Gesamtmasse 33.3 kg. In der Legacy-Reihenfolge ist Körper 0 das Becken; die Knöchel-Roll-Links sind 7 und 14, die Hände 28 und 38.

### 6.5 Zustands-Tensoren und ihre Views

`Humanoid_SMPLX.__init__` (`ultra/env/tasks/humanoid.py:82`) bildet wie früher Views, jetzt auf die Legacy-Tensoren:

| Attribut | Form | Bedeutung |
|---|---|---|
| `_humanoid_root_states` | (N, 13) | Roboter-Root, View auf Actor 0 jedes Env |
| `_target_states` | (N, 13) | Objekt, View auf Actor 1 jedes Env |
| `_dof_pos`, `_dof_vel` | (N, 29) | Gelenkzustand |
| `_rigid_body_pos/rot/vel/ang_vel` | (N, 39, 3 oder 4) | alle Roboter-Links |
| `_contact_forces` | (N, 39, 3) | Kontaktkraft pro Roboter-Link |
| `_tar_contact_forces` | (N, 3) | Kontaktkraft am Objekt (Index 39 im Env) |
| `dof_force_tensor` | (N, 29) | Gelenkmomente |

Eine Zuweisung wie `self._humanoid_root_states[env_ids, 0:3] = …` landet im Legacy-Tensor und wird mit dem nächsten `_set_…_indexed` in die Simulation geschrieben. Der umgekehrte Weg ist neu zu beachten: Jedes `_refresh_sim_tensors()` überschreibt die Legacy-Tensoren mit dem Simulationszustand. Was man hineingeschrieben, aber nicht mit `_set_…` abgeschickt hat, ist danach weg.

Aus den Body-Namen der YAML werden Index-Tensoren gebaut: `_key_body_ids` und `_contact_body_ids` (je 14 Links). Die Suche geht jetzt über `LEGACY_BODY_NAMES.index(name)` statt über ein Simulator-Handle. Gemessen: `[4, 5, 7, 11, 12, 14, 17, 30, 23, 24, 28, 33, 34, 38]`. Daneben stehen `_key_body_ids_gt` und `_contact_body_ids_gt` aus `keyIndex`/`contactIndex`; das sind die Indizes derselben 14 Punkte im 52-Körper-Skelett der Menschdaten.

### 6.6 Welche Isaac-Lab-Aufrufe wo stehen

Ausserhalb von `ultra/isaac/` ruft der Task-Code nur Methoden von `UltraBaseEnv` auf, mit zwei Ausnahmen (`root_physx_view` für Massen).

| Methode in `UltraBaseEnv` | Isaac-Lab-Aufruf | Benutzt von |
|---|---|---|
| `_refresh_sim_tensors` | `robot.data.*`, `object.data.root_state_w`, `*_contact.data.net_forces_w` | jeder Physikschritt, Reset |
| `_set_actor_root_state_indexed` | `write_root_state_to_sim` | Reset, Stösse, Datensatz abspielen |
| `_set_dof_state_indexed` | `write_joint_state_to_sim` | Reset |
| `_apply_dof_position_targets` | `robot.set_joint_position_target` | Stufe 1 und 2 |
| `_apply_dof_efforts` | `robot.set_joint_effort_target` | Teacher, Student |
| `_simulate` | `scene.write_data_to_sim()`, `sim.step(render=False)`, `scene.update(dt)` | Physikschleife |
| `_dof_limits` | `robot.data.joint_pos_limits` | Aktionsskalierung, Strafen |
| `_set_shape_materials` | `root_physx_view.set_material_properties` | Reibung Roboter, Material Objekt |
| `_set_gravity` | `sim.physics_sim_view.set_gravity` | Schwerkraft-Randomisierung |
| `_set_object_color` | `PreviewSurfaceCfg`, `bind_visual_material` | Kontaktanzeige beim Abspielen |
| `render` | `sim.render()` | Viewer |
| direkt: `robot.root_physx_view` | `set_masses`, `set_inertias`, `set_coms` | `Humanoid_G1._randomize_rigid_body_props` |
| direkt: `object.root_physx_view` | dasselbe | `UltraG1Retarget._setup_target_properties` |

### 6.7 Ein Policy-Schritt

```
PPO-Agent (UltraAgent.play_steps)
│  env_reset(done_indices)          Envs zurücksetzen, die im letzten Schritt fertig wurden
│  Aktion aus dem Netz
▼
UltraVecTask.step                   Aktion auf [-1, 1] begrenzen
▼
UltraBaseEnv.step                   ersetzt DirectRLEnv.step
├─ pre_physics_step                 Aktion speichern, letzte Werte merken, Aktionsverzögerung (nur mit DR)
├─ _physics_step                    render(), dann 17-mal:
│    ├─ Stellgrösse setzen          Positionsziel (Stufe 1) oder Moment (Teacher/Student)
│    ├─ _simulate                   write_data_to_sim → sim.step → scene.update
│    └─ _refresh_sim_tensors        Isaac Lab → Legacy-Tensoren
└─ post_physics_step
     ├─ progress_buf += 1           Referenzframe weiterzählen
     ├─ Stösse / Schwerkraft        nur mit DR
     ├─ _compute_hoi_observations   aktuellen Zustand im Referenzformat zusammenstellen
     ├─ _compute_observations       Eingabe für die Policy
     ├─ _compute_reward
     └─ _compute_reset              reset_buf und _terminate_buf setzen
```

`UltraBaseEnv.step` (`base_env.py:90`) gibt `obs_buf, rew_buf, reset_buf, extras` zurück, also das alte Vierer-Tupel und nicht das gymnasium-Fünfer-Tupel. Die `DirectRLEnv`-Haken `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones` werfen `NotImplementedError`; sie werden nie erreicht.

Der Grund für den eigenen Ablauf: `DirectRLEnv.step` setzt fertige Envs sofort selbst zurück. Hier markiert `post_physics_step` nur in `reset_buf`, welche Envs fertig sind. Der Agent liest das als `dones` und ruft zu Beginn des nächsten Schritts `env_reset` für genau diese Envs auf. So steht ihm die letzte Observation der Episode noch zur Verfügung, um den Wert des Folgezustands zu schätzen. `_terminate_buf` (in `extras["terminate"]`) unterscheidet Abbruch durch Scheitern von normalem Ende: Nur bei Abbruch setzt der Agent diesen Wert auf null (`ultra/learning/ultra_agent.py:318`).

`max_episode_length` ist in `DirectRLEnv` eine schreibgeschützte Property. `UltraBaseEnv` überdeckt sie mit einem Klassenattribut, damit die Tasks dort ihre Motion-Längen ablegen können.

### 6.8 Die zwei Regelungsarten

Beide laufen über denselben `ImplicitActuatorCfg` (`scene_cfg.py:109`); der Schalter `retargetPositionControl` entscheidet, ob er Gains bekommt.

**Positionsregelung, Stufe 1 und 2** (`retargetPositionControl: True`): `stiffness` und `damping` pro Gelenk aus `legacy_layout.py`.

```
Ziel = Mitte der Gelenkgrenzen + halber Gelenkbereich · Aktion      (_action_to_pd_targets)
```

Das Ziel wird einmal pro Policy-Schritt berechnet und in jedem der 17 Physikschritte gesetzt. PhysX regelt implizit. Eine Aktion von ±1 entspricht genau der Gelenkgrenze. Als `self.torques` dient `applied_torque`, also die Näherung aus 5.5, nicht ein gemessenes Moment.

**Drehmomentregelung, Teacher und Student**: `stiffness = damping = 0`, PhysX bekommt nur ein Moment.

```
Moment = kp · (3 · Aktion + q₀ − q) − kd · q̇          (Humanoid_SMPLX._compute_torques)
```

mit `q₀ = 0` und dem Faktor 3 aus `control.action_scale`. Das Moment wird auf 80 % des Effort-Limits begrenzt und in **jedem** Physikschritt aus dem frischen Gelenkzustand neu berechnet, also mit 1 kHz. Mit Motor-Randomisierung werden `kp` und `kd` pro Env und Gelenk mit einem Faktor aus [0.8, 1.2] multipliziert.

In beiden Fällen gelten Armatur und Effort-Limit aus `legacy_layout.py`; die Gelenkreibung ist 0. Gemessen in der Simulation: Knie-Steifigkeit 99.1, Armatur 0.0251, Effort-Limit 139.

Der Grund für die Zweiteilung ist derselbe wie früher: Das Retargeting soll nur saubere, physikalisch plausible Bewegungen erzeugen und nutzt dafür die idealisierte, stabile PD-Regelung des Simulators. Teacher und Student müssen auf den echten Roboter, dessen Low-Level-Regler genau dieses explizite PD-Gesetz rechnet; nur in dieser Form lassen sich Motorstärke und Verzögerung randomisieren.

### 6.9 Reset

`Humanoid_SMPLX._reset_envs` (`ultra/env/tasks/humanoid.py:144`):

1. `_reset_actors`: Werte in die Legacy-Views schreiben. Roboter über `_set_env_state`, Objekt über `_reset_target`.
2. `_reset_env_tensors`: `_set_actor_root_state_indexed` und `_set_dof_state_indexed` für die Roboter, danach `_set_actor_root_state_indexed` für die Objekte.
3. `_refresh_sim_tensors`
4. `_compute_observations(env_ids)`: erste Observation für die zurückgesetzten Envs.

Welche Motion ein Env bekommt und bei welchem Frame es startet, hängt von `stateInit` ab:

| `stateInit` | Motion | Startframe |
|---|---|---|
| `Start` | `env_id % Anzahl Motions` | 0 |
| `Random` | `env_id % Anzahl Motions` | zufällig |
| `Hybrid` | zufällig unter den Motions, die zum Objekt des Env passen | zufällig |

`progress_buf` ist der aktuelle Frame-Index in der Referenz, `start_times` der Startframe, `data_id` die Motion des Env.

### 6.10 Domain Randomization

Die Schalter stehen im `domain_rand:`-Block und wirken fast alle nur, wenn zusätzlich `domain_rand_general: True` gesetzt ist. In Stufe 1 ist alles aus.

| Was | Wie in Isaac Lab | Wann |
|---|---|---|
| Reibung Roboter | `set_material_properties`, ein Wert pro Env aus 64 Stufen | beim Aufbau |
| Masse und Schwerpunkt Torso | `root_physx_view.set_masses` / `set_inertias` / `set_coms` | beim Aufbau |
| Objektmasse, -schwerpunkt, -trägheit | dasselbe am Objekt; nur im Teacher-Task, dort immer | beim Aufbau |
| Motorstärke | Faktor auf `kp`/`kd` im eigenen PD-Gesetz | fest pro Env |
| Aktionsverzögerung | ältere Aktion aus `action_history_buf` | jeden Schritt |
| Stösse | Geschwindigkeit von Roboter und Objekt überschreiben, `write_root_state_to_sim` für alle | alle 4 s |
| Schwerkraft | `physics_sim_view.set_gravity` | alle 4 s |
| Rauschen auf Observations | in PyTorch | jeden Schritt |

Ohne Randomisierung setzt `_randomize_rigid_shape_props` die Reibung aller Roboter-Shapes auf 1.0 (gemessen), den Standard von Isaac Gym. Statische und dynamische Reibung bekommen denselben Wert, weil Isaac Gym nur einen Koeffizienten kannte.

Drei Dinge sind zu beachten. Die Schwerkraft gilt für die ganze Simulation: Eine Änderung trifft alle Envs gleichzeitig. Die Stösse treffen ebenfalls alle Envs im selben Schritt. Und das Objektmaterial wird anders als in der Isaac-Gym-Version nicht mehr randomisiert (siehe Kapitel 10).

`docs/install.md` nennt die Unterschiede in der Physik: Roll- und Drehreibung sowie Nachgiebigkeit der Objekte gibt es in PhysX 5 nicht; die konvexe Zerlegung stammt von PhysX 5 statt VHACD; mit Randomisierung haben Objekte fest 10 Hüllen statt zufällig 1 bis 10.

### 6.11 Anbindung an rl-games

`ultra/isaac/vec_task.py` enthält zwei Wrapper:

- `UltraVecTask`: baut die Spaces mit dem alten `gym` (rl-games 1.6 kennt `gymnasium` nicht), begrenzt Aktionen auf [−1, 1], reicht `reset(env_ids)` an den Task durch.
- `UltraDAggerVecTask` für die Student-Tasks: gibt statt `obs_buf` die Student-Observation zurück und daneben ein Dict mit Aktion, Mittelwert und normalisierter Observation des Teachers.

`RLGPUEnv` in `run.py` verpackt das als `IVecEnv` für rl-games. Die Agenten in `ultra/learning/` sind jetzt Unterklassen der installierten rl-games-Klassen (`a2c_continuous.A2CAgent`, `players.PpoPlayerContinuous`, `network_builder.A2CBuilder`, `ModelA2CContinuousLogStd`).

Eine Folge für Checkpoints: rl-games 1.6 hält die Eingangsnormalisierung im Modell, 1.1.4 hielt sie daneben. `load_checkpoint` in `ultra/learning/ultra_models.py` verschiebt die Statistik beim Laden an die richtige Stelle. Der mitgelieferte Teacher-Checkpoint lädt deshalb ohne Umwandlung.

### 6.12 Mehrere GPUs

`run.py` setzt `args.distributed = args.multi_gpu` vor dem Start des `AppLauncher`; dieser bindet dann jeden `torchrun`-Prozess an `cuda:LOCAL_RANK`. `create_rlgpu_env` verschiebt den Seed um den Rang. Jeder Prozess hat seine eigene vollständige Isaac-Sim-Instanz; `UltraAgent` mittelt über `torch.distributed`.

### 6.13 Viewer, Bilder, Datensatz ansehen

Ohne `--headless` zeigt Isaac Sim sein Fenster. `UltraViewer` (`ultra/isaac/viewer.py`) kapselt drei Dinge: Kamera nachführen (`sim.set_camera_view`), Debug-Punkte und -Linien (`isaacsim.util.debug_draw`) und Einzelbilder aus dem Viewport (`omni.replicator`). Alle übergebenen Positionen sind Env-lokal; der Ursprung wird dort addiert. `--save_images` schaltet `enable_cameras` ein, weil die Bildaufnahme auch ohne Fenster einen Renderer braucht.

Mit `--play_dataset` ruft der Player statt der Policy `play_dataset_step(t)` auf. Dabei wird der Referenzzustand Frame für Frame direkt in Root- und Gelenkzustand geschrieben. `scripts/play_dataset.sh` macht das mit dem Student-Task und den G1-Daten `[T, 630]`. Für die Menschdaten `[T, 591]` gibt es weiterhin keine passende Abspielfunktion: `UltraG1` erbt die Version aus `Ultra`, die 153 Gelenkwerte in die 29 Gelenke des G1 schreiben würde.

### 6.14 Das Prüfskript

`scripts/check_layout.py` prüft die Übersetzungsschicht ohne Datensatz. Es erzeugt den Teacher-Task mit einem synthetischen Clip, schreibt zufällige Root-Posen und Gelenkwinkel über die Legacy-Schnittstelle, macht einen Physikschritt ohne Schwerkraft und vergleicht die zurückgelesenen Körperposen mit der Vorwärtskinematik von MuJoCo auf `g1_29dof.xml`. Das deckt Körper- und Gelenkreihenfolge, Quaternion-Konvention und Env-Ursprünge ab.

Gemessen: `PASSED`, grösster Positionsfehler 6.9e-7 m, grösster Rotationsfehler 1.1e-3 rad.

---

## 7. Stufe 1: Retargeting-Policy trainieren

Konfiguration: `ultra/data/cfg/g1_retarget_smplx.yaml` (Env) und `ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml` (PPO). Observation, Reward und Abbruch rechnen auf den Legacy-Tensoren und sind gegenüber der Isaac-Gym-Version inhaltlich unverändert.

### 7.1 Referenz laden

`UltraG1._load_motion` (`ultra/env/tasks/ultra_g1.py:75`), pro Datei:

1. Ersten Frame verwerfen, dann alles linear von 30 auf 60 Hz interpolieren (`interp_time_series`). Auch die Quaternionen werden linear interpoliert, nicht per Slerp.
2. Root, Keypoints und Objektposition mit `sparseXYZMultiplier` pro Achse multiplizieren (Standard 1, 1, 1; beim Export die Augmentierung).
3. Geschwindigkeiten aus finiten Differenzen; Rotationsgeschwindigkeiten über die Exponentialdarstellung.
4. Über `keyIndex` die 14 Keypoints aus den 52 Körpern wählen. Nach der InterMimic-Reihenfolge sind das Hüfte, Knie und Knöchel je Seite, Torso, Kopf sowie Schulter, Ellbogen und Handgelenk je Seite. Sie werden den G1-Links aus `keyBodies` zugeordnet.
5. «Interaction Graph» berechnen: für jeden Keypoint der Vektor zum nächsten der 256 Objektpunkte, im Heading-Frame des Menschen.
6. Alles zu einem Referenzvektor mit 869 Werten pro Frame zusammensetzen (`hoi_data`), dazu ein kürzerer Vektor mit 332 Werten für den Reset (`hoi_refs`).
7. Den ersten Frame 30-mal voranstellen. In dieser halben Sekunde steht die Referenz still, und der Roboter hat Zeit, aus dem Stand in die Startpose zu kommen.

Alle Motions werden auf die Länge der längsten aufgefüllt und zu einem Tensor gestapelt. Gemessen mit synthetischen Clips von 80 Frames: Länge 187 = (80 − 2) · 2 + 1 + 30.

### 7.2 Reset

`UltraG1._set_env_state` (`ultra/env/tasks/ultra_g1.py:239`): Der Roboter startet nicht in der Referenzpose. Er steht in einer festen Standpose (`init_dof`, leicht gebeugte Knie und Ellbogen) an der mit 0.8 skalierten xy-Position der Referenz, auf Höhe 0.95 m (`initRootHeight`) und mit fester Gierrichtung. Das Objekt wird auf die Referenzpose gesetzt, xy mit 0.8 skaliert, Höhe unverändert.

In Isaac Lab kommt der Roboter tatsächlich nicht auf 0.95 m an, sondern steht nach dem ersten Physikschritt auf rund 0.80 m. Die Ursache ist das nicht normierte Start-Quaternion; Einzelheiten in Kapitel 10.

### 7.3 Skalierung

`scaling: 0.8` gleicht den Grössenunterschied zwischen Mensch und G1 aus. Überall, wo Roboter und Referenz in Weltkoordinaten verglichen werden, wird die Referenz mit 0.8 multipliziert (Positionen und Geschwindigkeiten, alle drei Achsen). Die Hand-Objekt-Vektoren des Interaction Graph werden nicht skaliert: Die Hand soll relativ zum Objekt dort sein, wo sie beim Menschen war. Passend dazu gibt es die Objekte in den Grössen 080 und 100.

### 7.4 Observation

1853 Werte (gemessen): zweimal derselbe Block von 926 Werten, einmal gegen den Referenzframe t+1 und einmal gegen t+16, plus ein Flag (`progress_buf >= 5`). Ein Block enthält:

| Teil | Werte | Inhalt |
|---|---|---|
| Eigener Zustand | 222 | Root-Höhe; Position, Rotation, Geschwindigkeit der 14 Key-Bodies im Heading-Frame; Kontakt-Flags |
| Differenz zur Referenz | 350 | Position, Rotation, Kontakt, Geschwindigkeit gegen die skalierte Referenz |
| Gelenke | 174 | Aktion, Gelenkposition und -geschwindigkeit, Momente, jeweils aktuelle und letzte Werte |
| Objekt | 21 | Objektgeschwindigkeit und Differenzen zur Referenz |
| Interaction Graph | 159 | Vektoren aller 39 Körper zum Objekt; Differenz zur Referenz für die 14 Keypoints |

«Heading-Frame» heisst: um die Hochachse so gedreht, dass der Roboter nach vorne schaut. Dadurch ist die Observation unabhängig von der Blickrichtung in der Welt. Die Vektoren des Interaction Graph werden als `Richtung · exp(−5 · Distanz)` kodiert, sodass nahe Körperteile ein starkes und ferne ein verschwindendes Signal geben.

Die Momente in der Observation sind in Isaac Lab die Näherung aus `applied_torque` (siehe 6.8). Das Rauschen auf der Objekt-Observation ist in Stufe 1 aus (`add_noise = not retargetPositionControl`).

### 7.5 Reward

`compute_humanoid_reward` (`ultra/env/tasks/humanoid_g1.py:345`). Die Terme werden multipliziert, nicht addiert: Ist ein Term schlecht, ist der ganze Reward schlecht. Über die Zeit wird zwischen drei Phasen überblendet (weiche Übergänge über Sigmoid-Funktionen von `progress_buf`):

| Phase | Zeitraum | Reward |
|---|---|---|
| Stehen | t < ~10 | Nähe zur Standpose |
| Annähern | dazwischen | `rb · ro · rig` |
| Tracking | t > ~20 | `rb · ro · rig · rcg` |

Die Terme:

- **`rb`, Körper**: Fusspositionen gegen die skalierte Referenz (Gewicht `p`), Richtungsvektoren von 9 Gliedmassen-Segmenten (Gewicht `r`), Geschwindigkeiten aller 14 Keypoints (Gewicht `pv`). Die Segmentrichtungen übertragen die Körperhaltung unabhängig von Gliedmassenlängen.
- **`ro`, Objekt**: Rotation (Gewicht `or`), Geschwindigkeit (Gewicht `opv`), Strafe auf Objektbeschleunigung. Die Objektposition ist fest mit Gewicht 0 abgeschaltet.
- **`rig`, Interaction Graph**: Vektoren beider Hände zu allen 256 Objektpunkten gegen dieselben Vektoren der Referenz, nach inverser Distanz gewichtet. Dieser Term überträgt die Hand-Objekt-Beziehung trotz anderer Körpergrösse.
- **`rcg`, Kontakt**: Handkontakt wie im Referenz-Label, die drei Handgelenk-Gelenke nahe null, Strafe auf grosse Kontaktkräfte ausserhalb der Füsse.

Das Ergebnis wird mit `exp(Glättungsstrafen / 10)` bzw. `/ 15` multipliziert. Die Glättungsstrafen (`env/tasks/smooth_rewards.py`, Gewichte im Block `smooth_rewards:`) sind Rumpfgeschwindigkeit, Gelenkgeschwindigkeit und -beschleunigung, Aktionsrate, Momente, Energie sowie Überschreiten von Gelenk- und Momentgrenzen.

Kontakt heisst überall: eine Komponente der Kontaktkraft des Sensors ist betragsmässig grösser als 0.1 N.

### 7.6 Abbruch

`compute_humanoid_reset` (`ultra/env/tasks/humanoid_g1.py:578`):

- Root unter 0.15 m, oder in den ersten 10 Schritten unter 0.5 m.
- Füsse im Mittel mehr als 0.5 m von der skalierten Referenz entfernt, oder das Objekt mehr als 0.5 m (gemessen an den ersten zwei der 256 Objektpunkte).
- Interaction-Graph-Fehler grösser als die Referenzdistanz selbst.
- Handkontakt fehlt mehr als 10 Schritte in Folge, obwohl die Referenz Kontakt hat.
- Ungültige Werte in der Observation lösen eine Exception aus.

Die referenzbezogenen Abbrüche gelten erst nach 30 Schritten. Regulär endet eine Episode am Ende der Motion oder nach `rolloutLength` Schritten.

### 7.7 PPO

| Einstellung | Wert |
|---|---|
| Netz | getrennte MLPs `[1024, 1024, 512]` für Actor und Critic, ReLU |
| Standardabweichung der Aktion | fest, nicht gelernt (log σ = −2.9) |
| Envs / Horizont | 4096 / 32 |
| Minibatch / Mini-Epochen | 16384 / 6 |
| Lernrate | 2e-5, konstant |
| γ / λ | 0.99 / 0.95 |
| Eingangsnormalisierung | ja (laufender Mittelwert und Varianz, im Modell) |
| Epochen | bis 50 000, Checkpoint alle 250 |

Checkpoints landen in `output/retarget_smplx/g1_retarget_smplx/nn/`. Weights & Biases ist standardmässig an; `WANDB_DISABLED=true` schaltet es ab.

---

## 8. Stufe 2: Export nach `[T, 630]`

`scripts/export_retarget_smplx.py` startet pro Clip und pro `--xyz`-Variante einen eigenen Prozess `run.py --test` mit überschriebener Konfiguration:

| Einstellung | Wert |
|---|---|
| `numEnvs` | 1 |
| `stateInit` | `Start` |
| `enableEarlyTermination` | `False` |
| `rolloutLength` | 100000 |
| `sparseXYZMultiplier` | die `--xyz`-Werte |
| `retargetExportPath` | Zieldatei |

Jeder Clip bezahlt damit den vollen Start von Isaac Sim. Bereits vorhandene Zieldateien werden übersprungen; das Log jedes Laufs liegt im Arbeitsverzeichnis neben dem Ausgabeordner. Als Erfolg gilt: Rückgabewert 0 und die Zieldatei existiert.

`UltraG1` zeichnet pro Policy-Schritt den Zustand von Env 0 auf (`_capture_retarget_frame`, `ultra/env/tasks/ultra_g1.py:40`), verwirft die 30 Stehframes und speichert am Episodenende:

| Spalten | Inhalt |
|---|---|
| 0:13 | Root-Zustand (Position, Quaternion, lineare und Winkelgeschwindigkeit) |
| 13:42 | Gelenkpositionen (29) |
| 42:71 | Gelenkgeschwindigkeiten (29) |
| 71:84 | Objektzustand (13) |
| 84:201 | Positionen der 39 Körper |
| 201:357 | Rotationen der 39 Körper |
| 357:474 | lineare Geschwindigkeiten der 39 Körper |
| 474:591 | Winkelgeschwindigkeiten der 39 Körper |
| 591:630 | Kontakt-Flag pro Körper |

Aufgezeichnet werden die Legacy-Tensoren. Die Datei hat deshalb dasselbe Format wie in der Isaac-Gym-Version: Quaternionen xyzw, Körper und Gelenke in Legacy-Reihenfolge, Positionen relativ zum Env. Der Teacher liest genau diese Indizes in `UltraG1Retarget._load_motion` wieder ein, und das veröffentlichte Archiv `OMOMO_retarget_aug` passt dazu.

Die Ausgabe wird als 60 Hz behandelt. Die Augmentierung entsteht durch Wiederholen mit anderen `--xyz`-Faktoren und mit der anderen Objektgrösse (`--asset-scale 100_100_100`): Dieselbe Policy fährt eine gestreckte oder gestauchte Referenz nach, und weil das Ergebnis aus der Physik kommt, bleibt es ausführbar.

---

## 9. Teacher und Student in Isaac Lab

### 9.1 Teacher (`UltraG1Retarget`)

Gleiche Basis wie das Retargeting, mit diesen Unterschieden:

| | Retargeting (`UltraG1`) | Teacher (`UltraG1Retarget`) |
|---|---|---|
| Referenz | Mensch, 14 Keypoints, skaliert | G1, alle 39 Körper, unskaliert |
| Aktuator | implizit mit Gains, Positionsziele | Gains 0, Momente aus eigenem PD-Gesetz |
| Objekt-Kollider | 5 Hüllen | 10 Hüllen |
| Objektmaterial (Reibung / Rückprall) | 0.5 / 0.6 | 0.6 / 0.05 |
| Observation | 1853 | 4052 |
| Start | Standpose, Frame 0 | Referenzpose, zufälliger Frame (`Hybrid`); Rauschen nur bei Start in Frame 0 |
| Domain Randomization | aus | an |
| Reward | drei Phasen, Glättung multiplikativ | `rb · ro · rig · rcg · 1.6`, Glättung additiv |
| Episodenlänge | bis 1000 Schritte | 300 Schritte |

Weitere Eigenheiten des Teachers:

- An jede Motion werden 20 Kopien des letzten Frames angehängt; dort soll der Roboter auf beiden Füssen stehen bleiben.
- 1 % der Envs sind «Stand still»: Der Referenzframe bleibt stehen.
- Pro Episode werden Objekt-Observation und Interaction-Graph-Merkmale zufällig behalten oder genullt (`obs_task_keep_prob`, `obs_ig_keep_prob`).
- Der Reward vergleicht zusätzlich Gelenkwinkel und -geschwindigkeiten direkt mit der Referenz, was erst mit G1-Referenzdaten möglich ist.
- Zusätzliche Strafen für den Gang: Fussorientierung, Rutschen, Stolpern, Fuss- und Knieabstand, Anheben des Schwungbeins.
- Objektmasse, -schwerpunkt und -trägheit werden pro Env einmal beim Aufbau gesetzt (`_setup_target_properties`, `ultra/env/tasks/ultra_g1_retarget.py:272`): Massenfaktor aus [0.15, 1.5], in jedem zehnten Env aus [0.001, 0.01]; die Trägheit wird mit demselben Faktor skaliert.

`ultra/run_teacher_inference.py` ersetzt die `run`-Methode des Players durch einen eigenen Rollout, startet dann `run.py` mit `g1_teacher_no_dr.yaml` und speichert Zustand, Referenz und Aktionen nach `output/teacher_inference/rollout.pt`.

### 9.2 Student (`UltraDistillObjV2Point`)

Der Task erbt vom Teacher-Task und lädt das Teacher-Netz direkt in die Umgebung (`env.teacherPolicy`, über `load_teacher_policy`). Er überschreibt `step()` (`ultra/env/tasks/ultra_g1_distill_obj_v2vae.py:1008`); nach der Physik passiert zusätzlich:

1. Die Teacher-Observation (4052) wird normalisiert und durch das Teacher-Netz geschickt.
2. Dessen Aktion und Mittelwert werden in `action_buf` und `mu_buf` abgelegt.
3. Die Student-Observation (1496) wird in `obs_buf_student` berechnet.

`UltraDAggerVecTask` gibt dem Agenten die Student-Observation und daneben die Teacher-Ausgabe als Lernziel zurück. Der Student sieht nur, was auf dem echten Roboter verfügbar ist: Propriozeption mit Verlauf über 10 Schritte, eine simulierte Punktwolke des Objekts aus Sicht der Kopfkamera (mit Rauschen, Ausfällen und Verdeckung), und Zielvorgaben. Welche Modalitäten sichtbar sind, wird pro Episode zufällig maskiert; die Wahrscheinlichkeiten sinken im Lauf des Trainings. Die Abbrüche wegen Abweichung von der Referenz sind abgeschaltet.

Die Punktwolke wird weiterhin in PyTorch aus Objektpose und Kamerapose gerechnet. Es gibt keinen Isaac-Lab-Kamerasensor in der Szene; die Student-Tasks benutzen von Isaac Lab nur das, was schon der Teacher benutzt, plus Debug-Zeichnen und Bildaufnahme über `UltraViewer`.

`UltraDistillObjV3RL` ergänzt eigene Rewards und Resets, um den Student danach mit RL weiterzutrainieren.

---

## 10. Auffälligkeiten und Stolpersteine

Punkte mit «gemessen» stammen aus den Probeläufen mit synthetischen Clips; die übrigen aus dem Lesen des Codes.

Nachtrag 01.10.2026, mit den 818 vorbereiteten OMOMO-Clips gemessen und behoben:

- **Speicher beim Laden der Motions.** `UltraG1._load_motion` hielt alle Zwischentensoren pro Clip und die aufgefüllten Kopien auf der GPU; mit 4096 Envs brach der Start mit `CUDA out of memory` ab (`ultra_g1.py`, `torch.stack` der `hoi_refs`, 856 MiB). Die Puffer werden jetzt auf der CPU zusammengesetzt und einmal auf die GPU kopiert; auf der GPU bleiben `hoi_data` (2.2 GiB) und `hoi_refs` (0.8 GiB). Das Ergebnis ist bis auf Rundung (≤ 2.4e-7 in den Interaction-Graph-Spalten) gleich.
- **4096 Envs passen auf 16 GB auch fürs Retargeting nicht**; das erste PPO-Update läuft in `CUDA out of memory`. Mit `--num_envs 2048`: Spitze 12.4 GB.
- **TorchScript machte jeden Reset minutenlang.** Die `@torch.jit.script`-Funktion `compute_humanoid_observations_max` wird mit torch 2.7 für jede neue Batchgrösse neu spezialisiert: gemessen 172 s für den Reset von 7 Envs, 207 s für 13 Envs. Im Training ändert sich die Zahl der zurückgesetzten Envs ständig; das ergab rund 300 Schritte/s. `ultra/run.py` und `ultra/run_teacher_inference.py` setzen jetzt vor dem Import von torch `PYTORCH_JIT=0` (überschreibbar). Danach: Reset 0.01 s, Training rund 5500 Schritte/s bei 2048 Envs, etwa 12 s pro Epoche.

- **Das Start-Quaternion ist nicht normiert, und in Isaac Lab hat das eine sichtbare Wirkung.** `UltraG1._set_env_state` schreibt `(0, 0, 1, −1)` mit Norm 1.41 (`ultra/env/tasks/ultra_g1.py:242`). Gemessen: Direkt nach dem Reset meldet die Simulation das Becken auf 0.95 m, den linken Knöchel aber auf 0.049 m, also 0.90 m darunter, obwohl die Beinkette in dieser Pose nur 0.75 m misst. Nach dem ersten Physikschritt ist das Quaternion normiert, der Knöchel unverändert, und das Becken liegt auf 0.798 m. Gegenprobe mit vorher normiertem Quaternion: Knöchel auf 0.201 m, der Roboter fällt frei aus 0.95 m und hat nach 170 ms den Boden noch nicht erreicht. Der Roboter startet in der Portierung also praktisch stehend auf 0.80 m statt mit einem Fall aus 0.95 m. Ob Isaac Gym sich gleich verhielt, ist nicht geprüft. Die erste Observation jeder Episode beruht auf dem inkonsistenten Zustand vor dem ersten Physikschritt.
- **Objekt und Motion passen in Stufe 1 oft nicht zusammen.** Bei `stateInit: Start` bekommt Env k die Motion `k % Anzahl Motions` (`ultra/env/tasks/ultra.py:336`), das Objekt aber `k % Anzahl Objekttypen` (`ultra/isaac/scene_cfg.py:156`). Gemessen mit drei Clips (eine Box, zwei Koffer) und 4 Envs: Objekt der Envs `[0, 1, 0, 1]`, Objekt der zugeteilten Motions `[0, 1, 1, 0]`. In den Envs 2 und 3 liegt ein anderes Objekt in der Simulation, als Reward und Observation annehmen. Die objektkonsistente Variante steht auskommentiert eine Zeile darüber. Der Teacher mit `Hybrid` ist nicht betroffen, der Export mit einem Env und einem Clip auch nicht.
- **Eine Null-Aktion ist im Retargeting keine Standpose.** Das Positionsziel für Aktion 0 ist die Mitte des Gelenkbereichs, gemessen z. B. Hüft-Roll ±1.22 rad und Knie 1.40 rad. Mit Null-Aktionen sackt der Roboter in 30 Schritten auf 0.46 m Beckenhöhe zusammen. Für einen schnellen Funktionstest taugen Null-Aktionen deshalb nicht.
- **Nach `simulation_app.close()` läuft kein Code mehr.** Gemessen: Der Aufruf beendet den Prozess mit Code 0. In `scripts/check_layout.py` wird das `sys.exit(0 if passed else 1)` dahinter nie erreicht; der Rückgabewert ist auch bei `FAILED` 0. Massgeblich ist die ausgegebene Zeile. Bei umgeleiteter Ausgabe kann sie zudem im Puffer verloren gehen (beobachtet bei `check_layout.py > datei`); `python -u` behebt das.
- **Das Objektmaterial wird nicht mehr randomisiert.** `g1_teacher.yaml` führt `obj_friction_range`, `obj_restitution_range`, `obj_inertia_range`, `obj_rolling_friction_range`, `obj_torsion_friction_range`, `obj_compliance_range`, `obj_rest_offset_range` und `obj_contact_offset_range` auf, und der Kommentar dort verweist auf eine Methode `randomize_physical_properties`. Im Code liest keine Stelle diese Schlüssel. Der Teacher setzt Reibung 0.6 und Rückprall 0.05 fest und randomisiert nur Masse, Schwerpunkt und (über den Massenfaktor) Trägheit. Die Isaac-Gym-Version randomisierte Reibung, Rückprall und Kontaktabstände beim Aufbau und bei jedem Reset.
- **Die «Momente» der Positionsregelung sind eine Näherung.** `self.torques` in Stufe 1 ist `applied_torque`: aus Ziel und Gelenkzustand vor dem Schritt gerechnet und auf das Effort-Limit begrenzt. Das betrifft die Momente in der Observation und die Strafen auf Moment, Energie und Momentgrenzen. Isaac Gym lieferte an dieser Stelle die vom Solver gemeldete Gelenkkraft.
- **Die Kontaktabstände des G1 kommen nicht aus der YAML.** `contact_offset` und `rest_offset` im `sim.physx`-Block wirken nur auf das Objekt. Für den Roboter stehen die Werte als Konstanten in `scripts/convert_assets.py` und in der USD; eine Änderung braucht eine neue Konvertierung.
- **Mehrere YAML-Einträge sind wirkungslos.** In Stufe 1 die Reward-Gewichte `ig`, `cg1`, `cg2`, `op` und `rv` (das Gewicht des Interaction Graph ist fest 5); überall `control.stiffness` und `control.damping` (die Gains stehen in `legacy_layout.py`), `sim.substeps`, `sim.physx.num_threads`, der `flex`-Block, `env.terminationHeight` (fest 0.15), `env.episodeLength`, `env.dataFPS` und `env.asset.assetRoot`. `action_scale` und `control_type` werden dagegen gelesen.
- **Die feste Startausrichtung setzt passende Daten voraus.** Eigene Clips müssten wie die InterMimic-Daten ausgerichtet sein, sonst startet der Roboter verdreht zur Referenz.
- **Training sieht nur die ersten 1000 Schritte pro Clip.** `rolloutLength: 1000` mit Start bei Frame 0 deckt etwa 17 s ab; längere Clips werden erst beim Export vollständig abgespielt.
- **Die Curricula sind von Anfang an abgeschlossen.** `common_step_counter` startet bei 320 000 (`ultra/env/tasks/humanoid.py:68`), die Schwellen für Stösse, Rauschen und Glättungsstrafen liegen bei 32 000 bis 160 000 Schritten.
- **Im Teacher-Reward stehen Indizes aus der 14-Keypoint-Zählung.** `left_foot_idx = 2`, `right_foot_idx = 5` und `feet_indices = [2, 5]` (`ultra/env/tasks/ultra_g1_retarget.py:1015` und `:1120`) waren in Stufe 1 die Knöchel. Im Teacher mit 39 Körpern zeigen dieselben Zahlen in `LEGACY_BODY_NAMES` auf `left_hip_pitch_link` und `left_knee_link`.
- **Die Envs überlappen räumlich.** `envSpacing: 1` bei Bewegungen über mehrere Meter. Physikalisch ist das folgenlos, im Viewer mit vielen Envs aber unübersichtlich.
- **Namen führen in die Irre.** `Humanoid_SMPLX` ist die generische Basisklasse, `UltraG1Retarget` der Teacher, `ballSize` die Objektskalierung, `amp_observation_space` ein Überbleibsel ohne Diskriminator, `run_distill.py` nur noch ein Alias, und `_humanoid_actor_ids` sind Actor-Indizes einer Simulation, die es so nicht mehr gibt.
- **Die Physik ist eine andere.** PhysX 5 aus Isaac Sim, andere konvexe Zerlegung, keine Roll- und Drehreibung. `docs/install.md` hält fest, dass Retargeting- und Tracking-Policies eventuell weiter trainiert werden müssen. Ob der mitgelieferte Teacher-Checkpoint in Isaac Lab so gut läuft wie in Isaac Gym, ist hier nicht geprüft.
