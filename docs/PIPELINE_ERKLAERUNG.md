# ULTRA: Aufbau des Repos, Retargeting und Simulation mit Isaac Gym

Diese Erklärung beruht auf dem Lesen des Codes. Ausgeführt wurde nichts: Isaac Gym und die Daten (`InterAct/`) lagen beim Schreiben nicht vor. Aussagen über Isaac Gym selbst (Kapitel 5) beschreiben das Verhalten von Isaac Gym Preview 4, wie es dokumentiert ist und wie der Code es benutzt; sie sind nicht gegen eine Installation geprüft.

Schwerpunkt: alles bis und mit Retargeting sowie alles, was Isaac Gym benutzt. Die MuJoCo-Skripte (`sim2sim_*.py`, `utils/obs*.py`) und die Netzarchitektur des Students werden nur gestreift.

## Inhalt

1. [Das Wichtigste vorweg](#1-das-wichtigste-vorweg)
2. [Aufbau des Repos](#2-aufbau-des-repos)
3. [Die Pipeline in fünf Stufen](#3-die-pipeline-in-fünf-stufen)
4. [Eingabedaten: `[T, 591]`](#4-eingabedaten-t-591)
5. [Wie Isaac Gym funktioniert](#5-wie-isaac-gym-funktioniert)
6. [Wie das Repo Isaac Gym benutzt](#6-wie-das-repo-isaac-gym-benutzt)
7. [Stufe 1: Retargeting-Policy trainieren](#7-stufe-1-retargeting-policy-trainieren)
8. [Stufe 2: Export nach `[T, 630]`](#8-stufe-2-export-nach-t-630)
9. [Teacher und Student in Isaac Gym](#9-teacher-und-student-in-isaac-gym)
10. [Auffälligkeiten und Stolpersteine](#10-auffälligkeiten-und-stolpersteine)

---

## 1. Das Wichtigste vorweg

- **Es gibt keine SMPL-X-Verarbeitung im Repo.** Die Pipeline startet bei fertig aufbereiteten Tensoren `[T, 591]` im InterMimic-Format. Das SMPL-X-Körpermodell wird nirgends geladen; die Umrechnung von SMPL-X-Parametern in diese Tensoren passiert ausserhalb (InterMimic / InterAct).
- **Das Retargeting ist kein IK-Verfahren, sondern eine RL-Policy in Isaac Gym.** Der G1 lernt per PPO, die auf 0.8 skalierten menschlichen Keypoints samt Objekt physikalisch nachzufahren. Der aufgezeichnete Simulationszustand dieses Rollouts ist der retargetete Datensatz `[T, 630]`.
- **Alle Trainingsstufen laufen in Isaac Gym** und teilen sich dieselbe Basisklasse für Szene, Zustands-Tensoren und Physikschleife. Sie unterscheiden sich in Referenzdaten, Observation, Reward und Regelungsart.

---

## 2. Aufbau des Repos

```
scripts/                     Einstiegspunkte pro Stufe (Shell + 2 Python-Skripte)
ultra/run.py                 Stufe 1–3: Retargeting + Teacher (rl_games-Runner)
ultra/run_distill.py         Stufe 4–5: Student (DAgger/VAE, dann RL-Finetuning)
ultra/run_teacher_inference.py   Teacher auf einem Clip ausrollen und speichern
ultra/utils/config.py        CLI-Argumente, YAML laden, gymapi.SimParams
ultra/utils/parse_task.py    Task-Klasse per Name instanziieren + VecTask-Wrapper
ultra/utils/torch_utils.py   Quaternion-Hilfsfunktionen (Heading, exp-map, 6D)
ultra/env/tasks/             alle Isaac-Gym-Umgebungen
ultra/learning/              PPO-Agent, Netze, Player (angepasstes rl_games)
ultra/data/cfg/              Env-YAMLs; train/rlg/ = PPO-YAMLs
ultra/data/assets/g1/        G1 mit 29 Freiheitsgraden als URDF (Isaac Gym) und XML (MuJoCo)
ultra/data/assets/objects/   Objekt-URDFs/-Meshes, Skalierung 080 und 100
ultra/weights/               mitgelieferter Teacher-Checkpoint
ultra/sim2sim_*.py, utils/obs*.py, export_jit.py   MuJoCo und Deployment, ohne Isaac Gym
```

### Klassenhierarchie der Tasks

| Klasse | Datei | Rolle |
|---|---|---|
| `BaseTask` | `env/tasks/base_task.py` | Sim und Viewer anlegen, Puffer, `step()`-Gerüst |
| `Humanoid_SMPLX` | `env/tasks/humanoid.py` | Zustands-Tensoren, Physikschleife, Reset, Basis-Observations |
| `Humanoid_G1` | `env/tasks/humanoid_g1.py` | G1-Asset, PD-Gains, Reward und Observations für Stufe 1 |
| `Ultra` | `env/tasks/ultra.py` | Motion-Dateien, Objekt-Assets, Reset aus der Referenz |
| `UltraG1` | `env/tasks/ultra_g1.py` | **Retargeting-Task** (Stufe 1 und 2) |
| `UltraG1Retarget` | `env/tasks/ultra_g1_retarget.py` | trotz des Namens der **Teacher** (Stufe 3) |
| `UltraDistillObjV2Point` | `env/tasks/ultra_g1_distill_obj_v2vae.py` | Student-Distillation (Stufe 4) |
| `UltraDistillObjV3RL` | `env/tasks/ultra_g1_distill_obj_v3rl.py` | Student-Finetuning mit RL (Stufe 5) |

`UltraG1(Humanoid_G1, Ultra)` und `UltraG1Retarget(Humanoid_G1, Ultra)` nutzen Mehrfachvererbung. Die Auflösungsreihenfolge ist

```
UltraG1 → Humanoid_G1 → Ultra → Humanoid_SMPLX → BaseTask
```

Das erklärt die Reihenfolge im Konstruktor: `Ultra.__init__` sammelt zuerst Motion-Dateien und Objektnamen, `Humanoid_SMPLX.__init__` → `BaseTask.__init__` baut die Simulation, danach lädt `Ultra.__init__` die Motions (`_load_motion`) und legt die Objekt-Tensoren an (`_build_target_tensors`).

### Aufrufkette beim Start

```
scripts/train_retarget_smplx.sh
  └─ python ultra/run.py --task UltraG1 --cfg_env … --cfg_train …
       ├─ get_args()            gymutil.parse_arguments + eigene Flags
       ├─ load_cfg()            beide YAMLs laden, CLI-Overrides anwenden
       └─ rl_games Runner       Agent "ultra" = UltraAgent (PPO)
            └─ create_rlgpu_env()
                 ├─ parse_sim_params()   → gymapi.SimParams
                 └─ parse_task()         → eval("UltraG1")(cfg, sim_params, …)
                                           verpackt in VecTaskPythonWrapper
```

`run.py` registriert die Umgebung unter dem Namen `rlgpu` bei rl_games und die eigenen Klassen für Agent, Player, Modell und Netz unter `ultra`.

---

## 3. Die Pipeline in fünf Stufen

| Stufe | Eingabe → Ausgabe | Einstieg | Task |
|---|---|---|---|
| 1 Retargeting | `[T, 591]` Mensch → Policy | `scripts/train_retarget_smplx.sh` | `UltraG1` |
| 2 Export | `[T, 591]` + Policy → `[T, 630]` G1 | `scripts/export_retarget_smplx.py` | `UltraG1` |
| 3 Teacher | `[T, 630]` → Tracking-Policy | `scripts/train_teacher.sh` | `UltraG1Retarget` |
| 4 Student | `[T, 630]` + Teacher → Student | `scripts/train_student.sh` | `UltraDistillObjV2Point` |
| 5 Finetuning | Student → zielgerichtete Policy | `scripts/train_finetune.sh` | `UltraDistillObjV3RL` |

Die Kommentare in den Shell-Skripten zählen anders als das README (`train_teacher.sh` nennt sich dort "Stage 2"). Diese Erklärung folgt der Zählung des README.

---

## 4. Eingabedaten: `[T, 591]`

Eine Datei ist ein Tensor mit einer Zeile pro Frame bei 30 fps. So liest `UltraG1._load_motion` (`ultra/env/tasks/ultra_g1.py:81`) die Spalten:

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

Der Dateiname trägt Information: `sub10_largebox_003_080_080_080.pt`. `Ultra.__init__` liest daraus den Objektnamen (zweites Feld) und die Asset-Skalierung (die letzten drei Felder) und bildet daraus den Asset-Namen `largebox_080_080_080`.

`scripts/prepare_retarget_smplx.py` rechnet nichts um. Es filtert die Rohdateien (`subNN_objekt_NNN.pt`) auf die vier Objekte mit vorhandenen Assets (`largebox`, `plasticbox`, `smallbox`, `suitcase`) und legt Symlinks mit angehängter Asset-Skalierung an.

---

## 5. Wie Isaac Gym funktioniert

Isaac Gym (Preview 4) ist NVIDIAs GPU-Physiksimulator für Reinforcement Learning. Die Grundidee: Tausende Kopien derselben Szene laufen parallel in einer einzigen PhysX-Simulation auf der GPU, und der Zustand aller Kopien liegt als PyTorch-Tensor auf derselben GPU wie das neuronale Netz. Zwischen Physik und Policy werden keine Daten über die CPU kopiert. Das Projekt ist eingefroren (Nachfolger ist Isaac Lab), deshalb die alten Abhängigkeiten: Python 3.8 und `numpy < 1.24`.

### 5.1 Die Python-Module

| Modul | Zweck |
|---|---|
| `isaacgym.gymapi` | Die eigentliche Schnittstelle: Sim, Assets, Envs, Actors, Eigenschaften, Viewer |
| `isaacgym.gymtorch` | Brücke zwischen Simulator-Puffern und PyTorch (`wrap_tensor`, `unwrap_tensor`) |
| `isaacgym.gymutil` | Hilfen: Argument-Parser, `parse_sim_config`, Debug-Geometrie |
| `isaacgym.torch_utils` | Quaternion-Mathematik auf Tensoren (`quat_rotate`, `quat_mul`, `to_torch`, …) |

`isaacgym` muss vor `torch` importiert werden. Deshalb steht in `run_teacher_inference.py` der scheinbar unnötige Import mit entsprechendem Kommentar.

### 5.2 Das Objektmodell

```
Gym            Singleton, gymapi.acquire_gym()
└─ Sim         eine PhysX-Szene auf einer GPU
   ├─ Ground   eine Bodenebene für alle Envs
   ├─ Asset    Bauplan, einmal geladen (URDF/MJCF): Körper, Gelenke, Kollisionsformen
   └─ Env      eine Kopie der Szene (hier 4096)
      └─ Actor Instanz eines Assets in einem Env
         ├─ Rigid Bodies    die Links
         ├─ DOFs            die beweglichen Gelenk-Freiheitsgrade
         └─ Rigid Shapes    Kollisionsgeometrie der Links
```

Wichtige Unterscheidung: Ein **Asset** wird einmal geladen und ist nur ein Bauplan. Ein **Actor** ist eine Instanz davon in einem Env. Manche Eigenschaften setzt man am Asset (gelten dann für alle danach erzeugten Actors), andere am Actor (pro Env unterschiedlich).

Die Envs sind keine getrennten Simulationen. Alle liegen in derselben PhysX-Szene und werden nur dadurch getrennt, dass jedes Env eine eigene **Kollisionsgruppe** hat (siehe 5.7).

### 5.3 Lebenszyklus

1. `gymapi.acquire_gym()`
2. `gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)`
3. `gym.add_ground(...)`
4. `gym.load_asset(...)` für jeden Bauplan
5. Pro Env: `gym.create_env(...)`, dann `gym.create_actor(...)` für jeden Actor, dazu Eigenschaften setzen
6. `gym.prepare_sim(sim)`: schliesst den Aufbau ab und alloziert die GPU-Puffer. Danach kann die Szene nicht mehr verändert werden, und erst danach funktioniert die Tensor-API.
7. `gym.acquire_*_tensor(sim)` und `gymtorch.wrap_tensor(...)`: Zustands-Tensoren holen
8. Schleife: Stellgrössen setzen → `gym.simulate(sim)` → `gym.fetch_results(sim, True)` → `gym.refresh_*_tensor(sim)`

Dass die Szene nach `prepare_sim` feststeht, hat eine praktische Folge: Anzahl und Art der Actors pro Env sind für die ganze Laufzeit fix. Welches Objekt in welchem Env steht, wird beim Aufbau entschieden und kann beim Reset nicht mehr geändert werden.

### 5.4 Die Tensor-API

Das ist der Kern. Der Simulator hält seinen Zustand in wenigen grossen, flachen Puffern, jeweils über **alle Envs hinweg**:

| Tensor | Form | Inhalt pro Zeile |
|---|---|---|
| Actor Root State | (Actors gesamt, 13) | Position 3, Quaternion xyzw 4, lineare Geschwindigkeit 3, Winkelgeschwindigkeit 3 |
| DOF State | (DOFs gesamt, 2) | Position, Geschwindigkeit |
| Rigid Body State | (Körper gesamt, 13) | wie Root State, aber für jeden Link |
| Net Contact Force | (Körper gesamt, 3) | Summe aller Kontaktkräfte auf den Körper |
| DOF Force | (DOFs gesamt) | am Gelenk wirkendes Moment |

Drei Operationen gehören dazu:

- **`acquire_*_tensor` + `gymtorch.wrap_tensor`** liefert einen PyTorch-Tensor, der denselben Speicher benutzt wie der Simulator-Puffer. Es wird nichts kopiert. Man ruft das einmal auf und behält den Tensor.
- **`refresh_*_tensor`** füllt den Puffer mit dem aktuellen Simulationszustand. Ohne Refresh zeigt der Tensor veraltete Werte.
- **`set_*_tensor`** schreibt in die Simulation zurück. In den gewrappten Tensor zu schreiben ändert nur den Puffer; wirksam wird es erst durch den `set_`-Aufruf. `gymtorch.unwrap_tensor` macht aus dem PyTorch-Tensor wieder das Handle, das die API erwartet.

Die Reihenfolge der Zeilen ist fest: Env für Env, darin Actor für Actor in der Reihenfolge des Anlegens. Weil alle Envs gleich aufgebaut sind, lässt sich der flache Tensor mit `.view(num_envs, pro_env, …)` umformen. Genau das macht das Repo.

Schreiben gibt es in zwei Varianten:

- `set_actor_root_state_tensor(sim, tensor)`: schreibt alle Actors.
- `set_actor_root_state_tensor_indexed(sim, tensor, indices, n)`: schreibt nur die Actors mit den angegebenen **globalen Actor-Indizes**. Übergeben wird trotzdem immer der ganze Puffer. Der globale Index eines Actors ist `env_id * actors_pro_env + lokaler_index`.

Für Gelenkzustände analog `set_dof_state_tensor_indexed`; die Indizes sind auch hier Actor-Indizes, nicht DOF-Indizes.

Eine Eigenheit, die für Resets wichtig ist: Root State und DOF State lassen sich setzen, der Rigid Body State nicht. Die Positionen der einzelnen Links ergeben sich erst wieder aus der Physik, also nach dem nächsten `simulate`. Unmittelbar nach einem Reset zeigt der Rigid-Body-Tensor für das zurückgesetzte Env deshalb noch den alten Zustand.

### 5.5 Simulationsschritt und Zeit

- `sim_params.dt` ist die Zeit, um die ein `simulate`-Aufruf die Welt vorrückt.
- `sim_params.substeps` teilt diesen Schritt intern weiter auf.
- Die Policy läuft langsamer als die Physik. Das Verhältnis heisst im Repo `controlFrequencyInv`: so viele `simulate`-Aufrufe pro Policy-Schritt.

`simulate` stösst die Berechnung an, `fetch_results(sim, True)` wartet auf das Ergebnis. Mit Viewer kommen `step_graphics` und `draw_viewer` dazu; diese sind von der Physik unabhängig.

### 5.6 Gelenkantriebe

Jeder DOF hat einen Antriebsmodus (`driveMode` in den DOF-Eigenschaften):

| Modus | Stellgrösse | Wer rechnet das Moment |
|---|---|---|
| `DOF_MODE_POS` | Zielposition über `set_dof_position_target_tensor` | PhysX: `stiffness · (Ziel − q) − damping · q̇` |
| `DOF_MODE_EFFORT` | Moment über `set_dof_actuation_force_tensor` | der eigene Code |
| `DOF_MODE_NONE` | keine | niemand, das Gelenk ist passiv |

Weitere DOF-Eigenschaften:

- `stiffness`, `damping`: Gains des internen PD-Reglers, nur im Positionsmodus wirksam.
- `effort`: maximales Moment.
- `armature`: zusätzliche Trägheit direkt am Gelenk. Modelliert die reflektierte Rotorträgheit des Getriebemotors und stabilisiert die Simulation bei steifen Reglern.
- `lower`, `upper`: Gelenkgrenzen.

Der interne PD-Regler im Positionsmodus wird vom Solver implizit mitgelöst und bleibt deshalb auch mit hohen Gains stabil. Im Effort-Modus gilt das berechnete Moment für einen ganzen Physikschritt als konstant; der Regler ist dann nur so gut wie die Rate, mit der man ihn neu berechnet.

### 5.7 Kollisionen

Drei Mechanismen bestimmen, was womit kollidiert:

- **Kollisionsgruppe** (Argument von `create_actor`): Nur Actors derselben Gruppe kollidieren miteinander. Gibt man jedem Env seine eigene Gruppe, sehen sich die Envs gegenseitig nicht. Die Bodenebene kollidiert mit allen.
- **Kollisionsfilter** (Bitmaske, am Actor und pro Shape): Zwei Shapes derselben Gruppe kollidieren nur, wenn das bitweise UND ihrer Filter null ist. Teilen sie ein Bit, wird die Kollision unterdrückt. Damit schaltet man Selbstkollisionen gezielt ab.
- **Konvexe Zerlegung (VHACD)**: PhysX rechnet auf der GPU mit konvexen Formen. Ein nicht konvexes Mesh wird beim Laden in mehrere konvexe Hüllen zerlegt, wenn `vhacd_enabled` gesetzt ist. `max_convex_hulls` und `resolution` steuern, wie genau.

Dazu kommen die Kontaktparameter von PhysX:

- `contact_offset`: Abstand, ab dem ein Kontakt erzeugt wird. Grösser heisst robuster gegen Durchdringen, aber mehr Kontaktpaare.
- `rest_offset`: Abstand, in dem Körper zur Ruhe kommen.
- `max_depenetration_velocity`: begrenzt, wie heftig eine Durchdringung korrigiert wird.
- `bounce_threshold_velocity`: unterhalb dieser Geschwindigkeit wird kein Rückprall berechnet.

**Aggregate** (`begin_aggregate` / `end_aggregate`) fassen die Actors eines Env für die Kollisionserkennung zu einer Einheit zusammen. Das spart Rechenzeit in der groben Kollisionsphase.

### 5.8 Solver

PhysX bietet zwei Solver; `solver_type: 1` ist TGS (Temporal Gauss-Seidel), der bei Gelenkketten mit Kontakt genauer ist als der ältere PGS. `num_position_iterations` und `num_velocity_iterations` legen fest, wie viele Iterationen der Solver pro Schritt macht.

### 5.9 Viewer und Headless

Ohne `--headless` öffnet `create_viewer` ein Fenster. Mit `--headless` wird kein Grafikgerät benutzt (`graphics_device_id = -1`), was für Training auf Servern nötig ist. Bilder aus dem Viewer lassen sich mit `write_viewer_image_to_file` speichern.

---

## 6. Wie das Repo Isaac Gym benutzt

### 6.1 Simulationsparameter

`parse_sim_params` (`ultra/utils/config.py:150`) setzt die Grundwerte, der `sim:`-Block der Env-YAML überschreibt sie über `gymutil.parse_sim_config`:

| Parameter | Wert | Herkunft |
|---|---|---|
| `dt` | 1 ms | `SIM_TIMESTEP` in `config.py` |
| `substeps` | 1 | YAML |
| Solver | TGS | YAML |
| Positions-/Geschwindigkeits-Iterationen | 4 / 1 | YAML |
| `contact_offset` / `rest_offset` | 0.02 / 0.0 | YAML |
| `bounce_threshold_velocity` | 0.2 | YAML |
| `max_depenetration_velocity` | 1.0 | YAML |
| `controlFrequencyInv` | 17 | YAML, `env:`-Block |
| Hochachse, Schwerkraft | z, −9.81 | `set_sim_params_up_axis` |

Ein Policy-Schritt sind also 17 Physikschritte zu 1 ms, zusammen 17 ms oder etwa 58.8 Hz. Die Referenzdaten werden als 60 Hz behandelt und pro Policy-Schritt um einen Frame weitergezählt. Die Abweichung von rund 2 % wird hingenommen.

### 6.2 Szenenaufbau

`BaseTask.__init__` ruft `create_sim()` der Unterklasse und danach `prepare_sim`. Der Aufbau im Einzelnen:

1. `Humanoid_SMPLX.create_sim` (`ultra/env/tasks/humanoid.py:163`): Sim anlegen, Bodenebene mit Reibung 1.0 und Rückprall 0.
2. `UltraG1._create_envs` → `Ultra._load_target_asset` (`ultra/env/tasks/ultra.py:246`): Für jeden vorkommenden Objekttyp wird das URDF geladen, mit VHACD und Dichte 25 (`objectDensity`). Die Masse ergibt sich aus Volumen mal Dichte. Zusätzlich lädt `trimesh` das OBJ und sampelt 256 Oberflächenpunkte relativ zum Schwerpunkt der Vertices. Diese Punkte (`self.object_points`) sind die Grundlage für alle Hand-Objekt-Distanzen in Reward und Observation; sie existieren nur in PyTorch, nicht in der Simulation.
3. `Humanoid_G1._create_envs` (`ultra/env/tasks/humanoid_g1.py:50`): G1-URDF laden, mit VHACD (höchstens 5 Hüllen pro Link). Dann pro Env `create_env`, `begin_aggregate`, `_build_env`, `end_aggregate`.
4. `_build_env` erzeugt pro Env zwei Actors: zuerst den Roboter (`Humanoid_G1._build_env`), dann das Objekt (`Ultra._build_target`). Die Reihenfolge legt die lokalen Actor-Indizes fest: Roboter 0, Objekt 1.

Beide Actors bekommen `env_id` als Kollisionsgruppe und Filter 0. Sie kollidieren also miteinander und mit dem Boden, nicht mit anderen Envs.

Der G1 hat laut Code 39 Rigid Bodies und 29 DOFs. Die DOF-Reihenfolge ist linkes Bein (6), rechtes Bein (6), Hüfte/Taille (3), linker Arm (7), rechter Arm (7). Die Reihenfolge der Bodies ist nicht die des URDF: Aus den fest codierten Indizes im Teacher (Hände bei 28 und 38, Knöchel bei 6, 7, 13, 14) folgt, dass Isaac Gym den Baum in der Tiefe durchläuft und Geschwister alphabetisch sortiert. Body 0 ist das Becken.

### 6.3 Eigenschaften des Roboters

`Humanoid_G1._build_env` (`ultra/env/tasks/humanoid_g1.py:185`) setzt pro Env:

- **DOF-Eigenschaften**: `effort` und `armature` für alle 29 DOFs aus fest codierten Listen. Im Retargeting-Modus zusätzlich `stiffness` und `damping` sowie `driveMode = DOF_MODE_POS`; sonst `DOF_MODE_EFFORT`. Die Werte im `control:`-Block der YAML (`stiffness`, `damping`) werden nicht benutzt.
- **Kollisionsfilter pro Shape**: rechts Knöchel 2, Knie 6, Hüfte 12; links 16, 48, 96. Knöchel und Knie teilen ein Bit, Knie und Hüfte ebenfalls, Knöchel und Hüfte nicht. Benachbarte Beinsegmente kollidieren also nicht miteinander, alles andere schon (auch linkes gegen rechtes Bein, Arme gegen Rumpf).
- **Kraftsensoren an den Gelenken**: `enable_actor_dof_force_sensors`, damit der DOF-Force-Tensor Werte liefert.
- **Domain Randomization beim Aufbau** (nur wenn eingeschaltet): Reibung der Shapes aus 64 Stufen, Zusatzmasse und verschobener Schwerpunkt am Torso.

Für das Objekt setzt `Ultra._build_target` Reibung 0.5 und Rückprall 0.6; der Teacher überschreibt das in seiner eigenen Version mit Reibung 0.6 und Rückprall 0.05 und randomisiert zusätzlich Masse, Schwerpunkt und Trägheit.

### 6.4 Zustands-Tensoren und ihre Views

`Humanoid_SMPLX.__init__` (`ultra/env/tasks/humanoid.py:87`) holt die fünf Tensoren aus 5.4 und bildet Views:

| Attribut | Form | Bedeutung |
|---|---|---|
| `_root_states` | (2·N, 13) | alle Actors, flach |
| `_humanoid_root_states` | (N, 13) | Roboter-Root, View auf Actor 0 jedes Env |
| `_target_states` | (N, 13) | Objekt, View auf Actor 1 jedes Env |
| `_dof_pos`, `_dof_vel` | (N, 29) | Gelenkzustand des Roboters |
| `_rigid_body_pos/rot/vel/ang_vel` | (N, 39, 3 oder 4) | alle Roboter-Links |
| `_contact_forces` | (N, 39, 3) | Kontaktkraft pro Roboter-Link |
| `_tar_contact_forces` | (N, 3) | Kontaktkraft am Objekt (Body-Index 39 im Env) |
| `dof_force_tensor` | (N, 29) | Gelenkmomente |

N ist die Anzahl Envs. Weil es Views sind, landet jede Zuweisung wie `self._humanoid_root_states[env_ids, 0:3] = …` direkt im Simulator-Puffer und wird mit dem nächsten `set_`-Aufruf wirksam.

Dazu die Indexlisten für das indizierte Schreiben: `_humanoid_actor_ids = 2 · env_id` und `_tar_actor_ids = 2 · env_id + 1`.

Aus den Body-Namen der YAML werden Index-Tensoren gebaut: `_key_body_ids` und `_contact_body_ids` (je 14 Links, über `find_actor_rigid_body_handle`). Daneben stehen `_key_body_ids_gt` und `_contact_body_ids_gt` aus `keyIndex`/`contactIndex`; das sind die Indizes derselben 14 Punkte im 52-Körper-Skelett der Menschdaten.

### 6.5 Ein Policy-Schritt

```
PPO-Agent (UltraAgent.play_steps)
│  env_reset(done_indices)          Envs zurücksetzen, die im letzten Schritt fertig wurden
│  Aktion aus dem Netz
▼
VecTaskPython.step                  Aktion auf [-1, 1] begrenzen
▼
BaseTask.step
├─ pre_physics_step                 Aktion speichern, letzte Werte merken, Aktionsverzögerung (nur mit DR)
├─ _physics_step                    17-mal:
│    ├─ Stellgrösse setzen          Positionsziel (Stufe 1) oder Moment (Teacher/Student)
│    ├─ gym.simulate
│    └─ _refresh_sim_tensors        fetch_results + alle refresh_*_tensor
└─ post_physics_step
     ├─ progress_buf += 1           Referenzframe weiterzählen
     ├─ Stösse / Schwerkraft        nur mit DR
     ├─ _compute_hoi_observations   aktuellen Zustand im Referenzformat zusammenstellen
     ├─ _compute_observations       Eingabe für die Policy
     ├─ _compute_reward
     └─ _compute_reset              reset_buf und _terminate_buf setzen
```

Dass die Tensoren nach jedem der 17 Physikschritte aufgefrischt werden, ist im Effort-Modus nötig, weil das Moment aus dem aktuellen Gelenkzustand berechnet wird. Im Positionsmodus dient es nur der Aufzeichnung (Gelenkgeschwindigkeiten und Momente pro Substep für die Glättungsstrafen).

`post_physics_step` setzt nicht selbst zurück. Es markiert nur in `reset_buf`, welche Envs fertig sind. Der Agent liest das als `dones` und ruft zu Beginn des nächsten Schritts `env_reset` für genau diese Envs auf. `_terminate_buf` unterscheidet Abbruch durch Scheitern von normalem Ende: Nur bei Abbruch setzt der Agent den Wert des Folgezustands auf null.

### 6.6 Die zwei Regelungsarten

**Positionsregelung, Stufe 1 und 2** (`retargetPositionControl: True`):

```
Ziel = Mitte der Gelenkgrenzen + halber Gelenkbereich · Aktion      (_action_to_pd_targets)
```

Das Ziel wird einmal pro Policy-Schritt berechnet und in jedem der 17 Substeps mit `set_dof_position_target_tensor` gesetzt. PhysX regelt mit den fest codierten Gains. Eine Aktion von ±1 entspricht genau der Gelenkgrenze. Als `self.torques` wird der DOF-Force-Tensor ausgelesen.

**Drehmomentregelung, Teacher und Student**:

```
Moment = kp · (3 · Aktion + q₀ − q) − kd · q̇          (Humanoid_SMPLX._compute_torques)
```

mit `q₀ = 0` und dem Faktor 3 aus `control.action_scale`. Das Moment wird auf 80 % des Effort-Limits begrenzt und in **jedem** Substep aus dem frischen Gelenkzustand neu berechnet, also mit 1 kHz. Mit Motor-Randomisierung werden `kp` und `kd` pro Env und Gelenk mit einem Faktor aus [0.8, 1.2] multipliziert.

Der Grund für die Zweiteilung: Das Retargeting soll nur saubere, physikalisch plausible Bewegungen erzeugen und nutzt dafür die idealisierte, stabile PD-Regelung des Simulators. Teacher und Student müssen auf den echten Roboter, dessen Low-Level-Regler genau dieses explizite PD-Gesetz rechnet; nur in dieser Form lassen sich Motorstärke und Verzögerung randomisieren.

### 6.7 Reset

`Humanoid_SMPLX._reset_envs` (`ultra/env/tasks/humanoid.py:188`):

1. `_reset_actors`: Werte in die Views schreiben. Roboter über `_set_env_state`, Objekt über `_reset_target`.
2. `_reset_env_tensors`: `set_actor_root_state_tensor_indexed` und `set_dof_state_tensor_indexed` für die Roboter, danach `set_actor_root_state_tensor_indexed` für die Objekte.
3. `_refresh_sim_tensors`
4. `_compute_observations(env_ids)`: erste Observation für die zurückgesetzten Envs.

Welche Motion ein Env bekommt und bei welchem Frame es startet, hängt von `stateInit` ab:

| `stateInit` | Motion | Startframe |
|---|---|---|
| `Start` | `env_id % Anzahl Motions` | 0 |
| `Random` | `env_id % Anzahl Motions` | zufällig |
| `Hybrid` | zufällig unter den Motions, die zum Objekt des Env passen | zufällig |

`progress_buf` ist der aktuelle Frame-Index in der Referenz, `start_times` der Startframe, `data_id` die Motion des Env.

### 6.8 Domain Randomization

Die Schalter stehen im `domain_rand:`-Block und wirken fast alle nur, wenn zusätzlich `domain_rand_general: True` gesetzt ist. Ausnahmen: Masse, Schwerpunkt und Trägheit des Objekts werden im Teacher-Task immer randomisiert, und das Rauschen auf den Observations hat mit `noise.add_noise` einen eigenen Schalter. In Stufe 1 ist alles aus.

| Was | Wie | Wann |
|---|---|---|
| Reibung Roboter | Shape-Eigenschaft am Asset, 64 Stufen | beim Aufbau |
| Masse und Schwerpunkt Torso | Rigid-Body-Eigenschaft | beim Aufbau |
| Objektmasse, -schwerpunkt, -trägheit | Rigid-Body-Eigenschaft | beim Aufbau |
| Objektreibung, -rückprall, Kontaktabstände | Shape-Eigenschaft | beim Aufbau und bei jedem Reset |
| Motorstärke | Faktor auf `kp`/`kd` im eigenen PD-Gesetz | fest pro Env |
| Aktionsverzögerung | ältere Aktion aus `action_history_buf` | jeden Schritt |
| Stösse | Geschwindigkeit von Roboter und Objekt überschreiben, `set_actor_root_state_tensor` | alle 4 s |
| Schwerkraft | `set_sim_params` | alle 4 s |
| Rauschen auf Observations | in PyTorch | jeden Schritt |

Zwei Dinge sind zu beachten. Die Schwerkraft ist ein Parameter der ganzen Simulation: Eine Änderung gilt für alle Envs gleichzeitig. Und die Stösse treffen ebenfalls alle Envs im selben Schritt.

### 6.9 Mehrere GPUs

`create_rlgpu_env` liest `RANK` und `LOCAL_RANK` von `torchrun`, bindet jeden Prozess an seine GPU und verschiebt den Seed um den Rang. Jeder Prozess hat seine eigene vollständige Isaac-Gym-Simulation; `UltraAgent` mittelt die Gradienten über `torch.distributed`.

### 6.10 Datensatz ansehen

Mit `--play_dataset` ruft der Player statt der Policy `play_dataset_step(t)` auf. Dabei wird der Referenzzustand Frame für Frame direkt in Root- und DOF-Tensoren geschrieben. `scripts/play_dataset.sh` macht das mit dem Student-Task und den G1-Daten `[T, 630]`. Für die Menschdaten `[T, 591]` gibt es keine passende Abspielfunktion: `UltraG1` erbt die Version aus `Ultra`, die 153 Gelenkwerte in die 29 DOFs des G1 schreiben würde.

---

## 7. Stufe 1: Retargeting-Policy trainieren

Konfiguration: `ultra/data/cfg/g1_retarget_smplx.yaml` (Env) und `ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml` (PPO).

### 7.1 Referenz laden

`UltraG1._load_motion` (`ultra/env/tasks/ultra_g1.py:81`), pro Datei:

1. Ersten Frame verwerfen, dann alles linear von 30 auf 60 Hz interpolieren (`interp_time_series`). Auch die Quaternionen werden linear interpoliert, nicht per Slerp.
2. Root, Keypoints und Objektposition mit `sparseXYZMultiplier` pro Achse multiplizieren (Standard 1, 1, 1; beim Export die Augmentierung).
3. Geschwindigkeiten aus finiten Differenzen; Rotationsgeschwindigkeiten über die Exponentialdarstellung.
4. Über `keyIndex` die 14 Keypoints aus den 52 Körpern wählen. Nach der InterMimic-Reihenfolge sind das Hüfte, Knie und Knöchel je Seite, Torso, Kopf sowie Schulter, Ellbogen und Handgelenk je Seite. Sie werden den G1-Links aus `keyBodies` zugeordnet.
5. "Interaction Graph" berechnen: für jeden Keypoint der Vektor zum nächsten der 256 Objektpunkte, im Heading-Frame des Menschen.
6. Alles zu einem Referenzvektor mit 869 Werten pro Frame zusammensetzen (`hoi_data`), dazu ein kürzerer Vektor für den Reset (`hoi_refs`).
7. Den ersten Frame 30-mal voranstellen. In dieser halben Sekunde steht die Referenz still, und der Roboter hat Zeit, aus dem Stand in die Startpose zu kommen.

Alle Motions werden auf die Länge der längsten aufgefüllt und zu einem Tensor gestapelt.

### 7.2 Reset

`UltraG1._set_env_state` (`ultra/env/tasks/ultra_g1.py:253`): Der Roboter startet nicht in der Referenzpose. Er steht in einer festen Standpose (`init_dof`, leicht gebeugte Knie und Ellbogen) an der mit 0.8 skalierten xy-Position der Referenz, auf Höhe 0.95 m (`initRootHeight`) und mit fester Gierrichtung. Das Objekt wird auf die Referenzpose gesetzt, xy mit 0.8 skaliert, Höhe unverändert.

### 7.3 Skalierung

`scaling: 0.8` gleicht den Grössenunterschied zwischen Mensch und G1 aus. Überall, wo Roboter und Referenz in Weltkoordinaten verglichen werden, wird die Referenz mit 0.8 multipliziert (Positionen und Geschwindigkeiten, alle drei Achsen). Die Hand-Objekt-Vektoren des Interaction Graph werden nicht skaliert: Die Hand soll relativ zum Objekt dort sein, wo sie beim Menschen war. Passend dazu gibt es die Objekte in den Grössen 080 und 100.

### 7.4 Observation

1853 Werte: zweimal derselbe Block von 926 Werten, einmal gegen den Referenzframe t+1 und einmal gegen t+16, plus ein Flag (`progress_buf >= 5`). Ein Block enthält:

| Teil | Werte | Inhalt |
|---|---|---|
| Eigener Zustand | 222 | Root-Höhe; Position, Rotation, Geschwindigkeit der 14 Key-Bodies im Heading-Frame; Kontakt-Flags |
| Differenz zur Referenz | 350 | Position, Rotation, Kontakt, Geschwindigkeit gegen die skalierte Referenz |
| Gelenke | 174 | Aktion, Gelenkposition und -geschwindigkeit, Momente, jeweils aktuelle und letzte Werte |
| Objekt | 21 | Objektgeschwindigkeit und Differenzen zur Referenz |
| Interaction Graph | 159 | Vektoren aller 39 Körper zum Objekt; Differenz zur Referenz für die 14 Keypoints |

"Heading-Frame" heisst: um die Hochachse so gedreht, dass der Roboter nach vorne schaut. Dadurch ist die Observation unabhängig von der Blickrichtung in der Welt. Die Vektoren des Interaction Graph werden als `Richtung · exp(−5 · Distanz)` kodiert, sodass nahe Körperteile ein starkes und ferne ein verschwindendes Signal geben.

### 7.5 Reward

`compute_humanoid_reward` (`ultra/env/tasks/humanoid_g1.py:516`). Die Terme werden multipliziert, nicht addiert: Ist ein Term schlecht, ist der ganze Reward schlecht. Über die Zeit wird zwischen drei Phasen überblendet (weiche Übergänge über Sigmoid-Funktionen von `progress_buf`):

| Phase | Zeitraum | Reward |
|---|---|---|
| Stehen | t < ~10 | Nähe zur Standpose |
| Annähern | dazwischen | `rb · ro · rig` |
| Tracking | t > ~20 | `rb · ro · rig · rcg` |

Die Terme:

- **`rb`, Körper**: Fusspositionen gegen die skalierte Referenz (Gewicht `p`), Richtungsvektoren von 9 Gliedmassen-Segmenten (Gewicht `r`), Geschwindigkeiten aller 14 Keypoints (Gewicht `pv`). Die Segmentrichtungen übertragen die Körperhaltung unabhängig von Gliedmassenlängen.
- **`ro`, Objekt**: Rotation (Gewicht `or`), Geschwindigkeit (Gewicht `opv`), Strafe auf Objektbeschleunigung. Die Objektposition ist fest mit Gewicht 0 abgeschaltet.
- **`rig`, Interaction Graph**: Vektoren beider Hände zu allen 256 Objektpunkten gegen dieselben Vektoren der Referenz, nach inverser Distanz gewichtet. Dieser Term überträgt die Hand-Objekt-Beziehung trotz anderer Körpergrösse.
- **`rcg`, Kontakt**: Handkontakt wie im Referenz-Label, die drei Handgelenk-DOFs nahe null, Strafe auf grosse Kontaktkräfte ausserhalb der Füsse.

Das Ergebnis wird mit `exp(Glättungsstrafen / 10)` bzw. `/ 15` multipliziert. Die Glättungsstrafen (`env/tasks/smooth_rewards.py`, Gewichte im Block `smooth_rewards:`) sind Rumpfgeschwindigkeit, Gelenkgeschwindigkeit und -beschleunigung, Aktionsrate, Momente, Energie sowie Überschreiten von Gelenk- und Momentgrenzen.

### 7.6 Abbruch

`compute_humanoid_reset` (`ultra/env/tasks/humanoid_g1.py:749`):

- Root unter 0.15 m, oder in den ersten 10 Schritten unter 0.5 m.
- Füsse im Mittel mehr als 0.5 m von der skalierten Referenz entfernt, oder das Objekt mehr als 0.5 m.
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
| Eingangsnormalisierung | ja (laufender Mittelwert und Varianz) |
| Epochen | bis 50 000, Checkpoint alle 250 |

Checkpoints landen in `output/retarget_smplx/g1_retarget_smplx/nn/`. Die Normalisierungsstatistik ist Teil des Checkpoints und wird beim Export wieder geladen.

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

Jeder Clip bezahlt damit den vollen Start von Isaac Gym. Bereits vorhandene Zieldateien werden übersprungen; das Log jedes Laufs liegt im Arbeitsverzeichnis neben dem Ausgabeordner.

`UltraG1` zeichnet pro Policy-Schritt den Zustand von Env 0 auf (`_capture_retarget_frame`, `ultra/env/tasks/ultra_g1.py:46`), verwirft die 30 Stehframes und speichert am Episodenende:

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

Die Ausgabe hat 60 Hz. Der Teacher liest genau diese Indizes in `UltraG1Retarget._load_motion` wieder ein.

Die Augmentierung entsteht durch Wiederholen mit anderen `--xyz`-Faktoren und mit der anderen Objektgrösse (`--asset-scale 100_100_100`): Dieselbe Policy fährt eine gestreckte oder gestauchte Referenz nach, und weil das Ergebnis aus der Physik kommt, bleibt es ausführbar.

---

## 9. Teacher und Student in Isaac Gym

### 9.1 Teacher (`UltraG1Retarget`)

Gleiche Basis wie das Retargeting, mit diesen Unterschieden:

| | Retargeting (`UltraG1`) | Teacher (`UltraG1Retarget`) |
|---|---|---|
| Referenz | Mensch, 14 Keypoints, skaliert | G1, alle 39 Körper, unskaliert |
| Regelung | Positionsziele, PhysX-PD | Drehmoment, eigenes PD-Gesetz |
| Observation | 1853 | 4052 |
| Start | Standpose, Frame 0 | Referenzpose, zufälliger Frame (`Hybrid`); Rauschen nur bei Start in Frame 0 |
| Domain Randomization | aus | alles an |
| Reward | drei Phasen, Glättung multiplikativ | `rb · ro · rig · rcg · 1.6`, Glättung additiv |
| Episodenlänge | bis 1000 Schritte | 300 Schritte |

Weitere Eigenheiten des Teachers:

- An jede Motion werden 20 Kopien des letzten Frames angehängt; dort soll der Roboter auf beiden Füssen stehen bleiben.
- 1 % der Envs sind "Stand still": Der Referenzframe bleibt stehen.
- Pro Episode werden Objekt-Observation und Interaction-Graph-Merkmale zufällig behalten oder genullt (`obs_task_keep_prob`, `obs_ig_keep_prob`).
- Der Reward vergleicht zusätzlich Gelenkwinkel und -geschwindigkeiten direkt mit der Referenz, was erst mit G1-Referenzdaten möglich ist.
- Zusätzliche Strafen für den Gang: Fussorientierung, Rutschen, Stolpern, Fuss- und Knieabstand, Anheben des Schwungbeins.

### 9.2 Student (`UltraDistillObjV2Point`)

Der Task erbt vom Teacher-Task und lädt das Teacher-Netz direkt in die Umgebung (`env.teacherPolicy`). In `step()` passiert nach der Physik zusätzlich:

1. Die Teacher-Observation (4052) wird normalisiert und durch das Teacher-Netz geschickt.
2. Dessen Aktion und Mittelwert werden in `action_buf` und `mu_buf` abgelegt.
3. Die Student-Observation (1496) wird in `obs_buf_student` berechnet.

`VecTaskDAggerWrapper` gibt dem Agenten die Student-Observation und daneben die Teacher-Ausgabe als Lernziel zurück. Der Student sieht nur, was auf dem echten Roboter verfügbar ist: Propriozeption mit Verlauf über 10 Schritte, eine simulierte Punktwolke des Objekts aus Sicht der Kopfkamera (mit Rauschen, Ausfällen und Verdeckung), und Zielvorgaben. Welche Modalitäten sichtbar sind, wird pro Episode zufällig maskiert; die Wahrscheinlichkeiten sinken im Lauf des Trainings. Die Abbrüche wegen Abweichung von der Referenz sind abgeschaltet.

`UltraDistillObjV3RL` ergänzt eigene Rewards und Resets, um den Student danach mit RL weiterzutrainieren.

---

## 10. Auffälligkeiten und Stolpersteine

Alle Punkte stammen aus dem Lesen des Codes und sind nicht durch einen Lauf bestätigt.

- **Objekt und Motion passen in Stufe 1 vermutlich oft nicht zusammen.** Bei `stateInit: Start` bekommt Env k die Motion `k % Anzahl Motions` (`ultra/env/tasks/ultra.py:391`), das Objekt-Asset aber `k % Anzahl Objekttypen` (`ultra/env/tasks/ultra.py:309`). Bei mehreren Objekttypen simuliert ein Env dann z. B. einen Koffer, während Reward und Observation mit den Punkten der grossen Box rechnen. Die objektkonsistente Variante steht auskommentiert eine Zeile darüber. Der Teacher mit `Hybrid` ist nicht betroffen, der Export mit einem Env und einem Clip auch nicht. Wegen 5.3 lässt sich das nicht beim Reset beheben, sondern nur über die Zuordnung der Motions.
- **Mehrere Reward-Gewichte in der YAML sind in Stufe 1 wirkungslos.** `ig`, `cg1`, `cg2`, `op` und `rv` werden in `compute_humanoid_reward` nicht gelesen; das Gewicht des Interaction Graph ist fest 5.
- **Die feste Startausrichtung setzt passende Daten voraus.** Eigene Clips müssten wie die InterMimic-Daten ausgerichtet sein, sonst startet der Roboter verdreht zur Referenz. Das Start-Quaternion `(0, 0, 1, −1)` ist zudem nicht normiert.
- **Training sieht nur die ersten 1000 Schritte pro Clip.** `rolloutLength: 1000` mit Start bei Frame 0 deckt etwa 17 s ab; längere Clips werden erst beim Export vollständig abgespielt.
- **Die erste Observation nach einem Reset beruht auf alten Link-Positionen.** Die Observation wird aus dem Rigid-Body-Tensor berechnet, der erst nach dem nächsten Physikschritt den neuen Zustand zeigt (siehe 5.4). Betroffen ist ein Schritt pro Episode.
- **Die Curricula sind von Anfang an abgeschlossen.** `common_step_counter` startet bei 320 000 (`ultra/env/tasks/humanoid.py:75`), die Schwellen für Stösse, Rauschen und Glättungsstrafen liegen bei 32 000 bis 160 000 Schritten.
- **Im Teacher-Reward stehen Indizes aus der 14-Keypoint-Zählung.** `left_foot_idx = 2`, `right_foot_idx = 5` und `feet_indices = [2, 5]` (`ultra/env/tasks/ultra_g1_retarget.py:1111` und `:1216`) waren in Stufe 1 die Knöchel. Im Teacher mit 39 Körpern zeigen dieselben Zahlen nach der Body-Reihenfolge aus 6.2 auf Hüft- und Knie-Links.
- **Die Gelenkgains stehen im Code, nicht in der YAML.** `stiffness` und `damping` im `control:`-Block sind Altlasten; `action_scale` und `control_type` werden dagegen gelesen.
- **Namen führen in die Irre.** `Humanoid_SMPLX` ist die generische Basisklasse, `UltraG1Retarget` der Teacher, `ballSize` die Objektskalierung, und `amp_observation_space` ein Überbleibsel ohne Diskriminator.
