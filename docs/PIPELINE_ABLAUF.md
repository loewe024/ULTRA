# ULTRA: Ablauf durch den Code, Stufe für Stufe

Mermaid-Diagramme zum Stand von `feature/isaaclab-port` (inklusive der Fixes vom 01./02.10.2026). Die Erklärung im Fliesstext steht in [`PIPELINE_ERKLAERUNG_ISAACLAB.md`](PIPELINE_ERKLAERUNG_ISAACLAB.md).

Alle Befehle laufen aus dem Repo-Root im Conda-Env `ultra` (`OMNI_KIT_ACCEPT_EULA=YES` setzen). Auf einer 16-GB-GPU hängt man an jeden Trainingsbefehl `--num_envs 2048` an.

Lesehilfe: Kästen sind Dateien bzw. Funktionen (`datei.py: Klasse.methode`), Pfeile die Aufrufreihenfolge. Gestrichelte Pfeile sind Daten, die eine Stufe für die nächste schreibt.

## Inhalt

1. [Übersicht der Stufen](#1-übersicht-der-stufen)
2. [Gemeinsamer Start in `run.py`](#2-gemeinsamer-start-in-runpy)
3. [Aufbau einer Umgebung](#3-aufbau-einer-umgebung)
4. [Ein Policy-Schritt](#4-ein-policy-schritt)
5. [Stufe 0: Assets konvertieren](#5-stufe-0-assets-konvertieren)
6. [Stufe 1: Retargeting trainieren](#6-stufe-1-retargeting-trainieren)
7. [Stufe 2: Export nach `[T, 630]`](#7-stufe-2-export-nach-t-630)
8. [Stufe 3: Teacher trainieren](#8-stufe-3-teacher-trainieren)
9. [Stufe 4: Student destillieren](#9-stufe-4-student-destillieren)
10. [Stufe 5: Student finetunen](#10-stufe-5-student-finetunen)
11. [Abspielen und Inferenz](#11-abspielen-und-inferenz)

---

## 1. Übersicht der Stufen

```mermaid
flowchart TD
    raw[("InterAct/OMOMO_retarget/*.pt<br/>Mensch, [T, 591], 30 fps")]
    urdf[("ultra/data/assets<br/>G1-URDF, Objekt-OBJ")]

    s0["Stufe 0: python scripts/convert_assets.py"]
    usd[("ultra/data/assets/usd/*.usd")]

    s1["Stufe 1: scripts/train_retarget_smplx.sh<br/>Task UltraG1"]
    ck1[("output/retarget_smplx/g1_retarget_smplx/nn/<br/>g1_retarget_smplx.pth")]

    s2["Stufe 2: python scripts/export_retarget_smplx.py<br/>Task UltraG1, --test"]
    aug[("G1-Bewegungen [T, 630], 60 Hz<br/>z. B. InterAct/OMOMO_retarget_aug/")]

    s3["Stufe 3: scripts/train_teacher.sh<br/>Task UltraG1Retarget"]
    ck3[("output/teacher/g1_teacher/nn/g1_teacher.pth<br/>oder ultra/weights/teacher_ultra_inference.pth")]

    s4["Stufe 4: scripts/train_student.sh<br/>Task UltraDistillObjV2Point"]
    ck4[("output/student/g1_student_vae/nn/g1_student_vae.pth")]

    s5["Stufe 5: scripts/train_finetune.sh student.pth<br/>Task UltraDistillObjV3RL"]
    ck5[("output/finetune/g1_student_finetune/nn/g1_student_finetune.pth")]

    deploy["Deployment / MuJoCo<br/>scripts/export_jit.sh, scripts/sim2sim_student.sh"]

    urdf --> s0 --> usd
    usd -.-> s1 & s2 & s3 & s4 & s5
    raw --> s1 --> ck1
    raw --> s2
    ck1 --> s2 --> aug
    aug --> s3 --> ck3
    aug --> s4
    ck3 -->|"env.teacherPolicy"| s4 --> ck4
    ck4 --> s5
    ck3 --> s5
    aug --> s5 --> ck5
    ck4 & ck5 --> deploy
```

| Stufe | Start | Task-Klasse | Agent |
|---|---|---|---|
| 0 | `python scripts/convert_assets.py [--force]` | – | – |
| 1 | `scripts/train_retarget_smplx.sh --num_envs 2048` | `UltraG1` | `UltraAgent` (PPO) |
| 2 | `python scripts/export_retarget_smplx.py --input-dir … --checkpoint … --output-dir … --xyz 1 1 1` | `UltraG1` | `UltraPlayerContinuous` |
| 3 | `scripts/train_teacher.sh --num_envs 2048` | `UltraG1Retarget` | `UltraAgent` (PPO) |
| 4 | `scripts/train_student.sh --num_envs 2048` | `UltraDistillObjV2Point` | `UltraAgentDistill` (DAgger/VAE) |
| 5 | `scripts/train_finetune.sh <student.pth> --num_envs 2048` | `UltraDistillObjV3RL` | `UltraAgentDistill` (Distillation + PPO) |

Stufe 3 ist optional, wenn man den mitgelieferten Teacher `ultra/weights/teacher_ultra_inference.pth` nimmt; Stufen 1 und 2 sind optional, wenn man das veröffentlichte Archiv `OMOMO_retarget_aug` nimmt.

---

## 2. Gemeinsamer Start in `run.py`

Alle Stufen ausser 0 laufen durch `ultra/run.py` (`run_distill.py` ist nur ein Alias darauf). Die Shell-Skripte unterscheiden sich nur in `--task`, den beiden YAML-Dateien und `--output_path`.

```mermaid
flowchart TD
    sh["scripts/*.sh<br/>python ultra/run.py --task ... --cfg_env ... --cfg_train ..."]

    subgraph start ["ultra/run.py, Modulebene"]
        jit["os.environ PYTORCH_JIT=0<br/>vor jedem torch-Import"]
        argp["utils/config.py: get_args_parser<br/>+ AppLauncher.add_app_launcher_args"]
        app["AppLauncher(args)<br/>startet Isaac Sim / Kit"]
        fin["utils/config.py: finalize_args<br/>--test setzt train=False"]
        imp["Imports: rl_games, torch,<br/>isaac.vec_task, learning.ultra_models"]
        reg["vecenv.register('RLGPU')<br/>env_configurations.register('rlgpu')"]
        jit --> argp --> app --> fin --> imp --> reg
    end

    subgraph mainf ["ultra/run.py: main"]
        lc["utils/config.py: load_cfg<br/>Env-YAML + PPO-YAML laden,<br/>--num_envs, --checkpoint, --max_iterations"]
        ov["CLI-Overrides ins cfg schreiben<br/>--motion_file, --resume_from, train_dir = --output_path"]
        wb["wandb.init, ausser WANDB_DISABLED=true"]
        bar["build_alg_runner<br/>Agent, Player, Netz je nach Task wählen"]
        rl["Runner.load(cfg_train)<br/>Runner.reset()"]
        rr["Runner.run(vargs)"]
        lc --> ov --> wb --> bar --> rl --> rr
    end

    sh --> jit
    reg --> lc

    rr -->|"train"| tr["Runner.run_train: Agent anlegen<br/>A2CBase.__init__ → vecenv.create_vec_env"]
    rr -->|"--test"| pl["Runner.run_play: Player anlegen<br/>BasePlayer.__init__ → create_env"]
    tr --> env["run.py: RLGPUEnv.__init__"]
    env --> cre["run.py: create_rlgpu_env<br/>Seed + Rang, cuda:LOCAL_RANK"]
    pl -->|"direkt, ohne RLGPUEnv"| cre
    cre --> pt["utils/parse_task.py: parse_task<br/>(Kapitel 3)"]
    pt --> rst["nur Training: RLGPUEnv.reset()<br/>erster Reset aller Envs"]
    pt --> player["Player.restore(checkpoint)<br/>Player.run()"]
    rst --> agent["Agent.train()<br/>(Stufen 1, 3, 4, 5)"]
    agent --> close["task.close()<br/>simulation_app.close() beendet den Prozess"]
    player --> close
```

`build_alg_runner` wählt die Klassen nach Task:

```mermaid
flowchart LR
    t{"resolve_task(--task)"}
    t -->|"UltraG1, UltraG1Retarget"| a1["learning/ultra_agent.py: UltraAgent<br/>learning/ultra_players.py: UltraPlayerContinuous<br/>learning/ultra_network_builder.py: UltraBuilder"]
    t -->|"UltraDistillObjV2Point"| a2["learning/ultra_agent_distill_vae.py: UltraAgentDistill<br/>learning/ultra_players_distill.py: UltraPlayerContinuousDistill<br/>learning/ultra_network_builder_obj_v2.py: UltraBuilder"]
    t -->|"UltraDistillObjV3RL"| a3["learning/ultra_agent_distill_vae_rl.py: UltraAgentDistill<br/>learning/ultra_players_distill.py: UltraPlayerContinuousDistill<br/>learning/ultra_network_builder_obj_v3.py: UltraBuilder"]
    a1 & a2 & a3 --> m["learning/ultra_models.py: ModelUltraContinuous<br/>alle unter dem Namen 'ultra' registriert"]
```

---

## 3. Aufbau einer Umgebung

`parse_task` erzeugt die Isaac-Lab-Umgebung. Der Konstruktor läuft durch die Mehrfachvererbung `UltraG1 → Humanoid_G1 → Ultra → Humanoid_SMPLX → UltraBaseEnv → DirectRLEnv`; jede Klasse macht einen Teil vor und einen Teil nach `super().__init__`. Für Teacher und Student ist die Kette gleich, nur steht `UltraG1Retarget` bzw. die Student-Klasse vorne.

```mermaid
flowchart TD
    pt["utils/parse_task.py: parse_task"]
    pt --> rt["resolve_task, register_tasks<br/>gymnasium-IDs registrieren"]
    rt --> mk["isaac/scene_cfg.py: make_env_cfg<br/>YAML-Dict → UltraEnvCfg,<br/>build_sim_cfg: dt = 1 ms, decimation = 17, PhysX"]
    mk --> gm["gym.make(id, cfg) → Task-Klasse"]

    subgraph ctor ["Konstruktor, von aussen nach innen"]
        c1["env/tasks/ultra_g1.py: UltraG1.__init__"]
        c2["env/tasks/humanoid_g1.py: Humanoid_G1.__init__<br/>keyIndex, contactIndex"]
        c3["env/tasks/ultra.py: Ultra.__init__<br/>Motion-Dateien auflisten,<br/>Objektnamen aus Dateinamen, object_id"]
        c4["env/tasks/humanoid.py: Humanoid_SMPLX.__init__<br/>YAML lesen, numObs, numActions"]
        c5["isaac/base_env.py: UltraBaseEnv.__init__"]
        c1 --> c2 --> c3 --> c4 --> c5
    end

    gm --> c1

    subgraph base ["UltraBaseEnv.__init__"]
        b1["scene_cfg.py: finalize_env_cfg<br/>build_robot_cfg: G1-USD + ImplicitActuatorCfg<br/>build_object_cfg: Objekt-USDs, Env i → Objekt i % n<br/>ContactSensorCfg, TerrainImporterCfg"]
        b2["DirectRLEnv.__init__<br/>SimulationContext, InteractiveScene klonen,<br/>_setup_scene (Licht), sim.reset()"]
        b3["_build_index_maps<br/>Legacy-Reihenfolge ↔ Isaac Lab<br/>(isaac/legacy_layout.py)"]
        b4["_allocate_state_tensors<br/>_root_states, _dof_state, _rigid_body_state, ..."]
        b5["_allocate_task_buffers<br/>obs_buf, rew_buf, reset_buf, progress_buf"]
        b6["_check_object_assignment"]
        b7["UltraViewer, nur ohne --headless"]
        b8["_setup_env_properties"]
        b9["_refresh_sim_tensors"]
        b1 --> b2 --> b3 --> b4 --> b5 --> b6 --> b7 --> b8 --> b9
    end

    c5 --> b1

    subgraph props ["UltraG1._setup_env_properties"]
        p1["Ultra._load_target_asset<br/>trimesh: 256 Oberflächenpunkte pro Objekt"]
        p2["Humanoid_G1._setup_env_properties<br/>Fuss/Knie/Torso-Indizes, Gelenkgrenzen,<br/>PD-Gains, Reibung, Torso-Masse"]
        p3["Ultra._setup_target_properties<br/>Objektmaterial"]
        p1 --> p2 --> p3
    end

    b8 --> p1

    subgraph back ["zurück aus super().__init__, von innen nach aussen"]
        r1["Humanoid_SMPLX: Views auf die Legacy-Tensoren,<br/>_key_body_ids, _contact_body_ids"]
        r2["Ultra: _load_motion (Stufe 1: UltraG1._load_motion),<br/>_build_target_tensors"]
        r3["Humanoid_G1: Aktionspuffer, SmoothRewards"]
        r4["UltraG1: scaling, initRootHeight, init_dof"]
        r1 --> r2 --> r3 --> r4
    end

    b9 --> r1
    r4 --> wr["isaac/vec_task.py: UltraVecTask<br/>(Student-Tasks: UltraDAggerVecTask)"]
```

---

## 4. Ein Policy-Schritt

Gleich für alle Stufen; die Unterschiede stecken in den Methoden, die die Task-Klassen überschreiben.

```mermaid
flowchart TD
    ag["Agent.play_steps, Schleife über horizon_length"]
    ag --> er["env_reset(done_indices)<br/>→ RLGPUEnv.reset → UltraVecTask.reset → task.reset"]
    er --> rs["Humanoid_SMPLX._reset_envs, nur fertige Envs"]

    subgraph reset ["Reset"]
        ra["_reset_actors<br/>Motion und Startframe wählen (stateInit),<br/>_set_env_state: Roboter, _reset_target: Objekt"]
        rb["_reset_env_tensors<br/>base_env: _set_actor_root_state_indexed,<br/>_set_dof_state_indexed → write_*_to_sim"]
        rc["_refresh_sim_tensors"]
        rd["_compute_observations(env_ids)"]
        ra --> rb --> rc --> rd
    end

    rs --> ra
    rd --> act["get_action_values: Netz → Aktion"]
    act --> es["env_step → UltraVecTask.step<br/>Aktion auf [-1, 1] begrenzen"]
    es --> st["isaac/base_env.py: UltraBaseEnv.step"]

    st --> pre["pre_physics_step<br/>Aktion speichern, Aktionsverzögerung (DR)"]
    pre --> phy["Humanoid_SMPLX._physics_step"]

    subgraph substeps ["17 x pro Policy-Schritt"]
        ctl{"retargetPositionControl?"}
        pos["_apply_dof_position_targets<br/>Ziel aus _action_to_pd_targets"]
        tor["_compute_torques: kp·(3a − q) − kd·q̇<br/>_apply_dof_efforts"]
        sim["base_env: _simulate<br/>write_data_to_sim → sim.step → scene.update"]
        ref["_refresh_sim_tensors<br/>Isaac Lab → Legacy-Tensoren"]
        ctl -->|"ja (Stufe 1, 2)"| pos --> sim
        ctl -->|"nein (Stufe 3, 4, 5)"| tor --> sim
        sim --> ref
    end

    phy --> ctl
    ref --> post["post_physics_step"]

    subgraph postp ["post_physics_step"]
        q1["progress_buf += 1"]
        q2["Stösse, Schwerkraft (nur mit DR)"]
        q3["_compute_hoi_observations"]
        q4["_compute_observations"]
        q5["_compute_reward"]
        q6["_compute_reset → reset_buf, _terminate_buf"]
        q1 --> q2 --> q3 --> q4 --> q5 --> q6
    end

    post --> q1
    q6 --> back["obs, reward, done, extras zurück an den Agenten"]
    back --> ag
```

---

## 5. Stufe 0: Assets konvertieren

**Start:** `python scripts/convert_assets.py` (einmalig; `--force` nach Änderungen an URDF oder Meshes).

```mermaid
flowchart TD
    s["scripts/convert_assets.py"]
    s --> al["AppLauncher, headless"]
    al --> g["convert_g1"]
    g --> uc["isaaclab UrdfConverter<br/>g1_29dof.urdf, merge_fixed_joints=False,<br/>convex_decomposition, self_collision"]
    uc --> pp["postprocess_g1 (pxr)"]
    pp --> pp1["masselose Links: 1 g"]
    pp --> pp2["Physik-Layer: contact_offset 0.02,<br/>Zerlegung 5 Hüllen"]
    pp --> pp3["FilteredPairsAPI: Bein-Kollisionsfilter"]
    pp1 & pp2 & pp3 --> gu[("usd/g1/g1_29dof.usd")]

    al --> o["convert_objects"]
    o --> mc["für jedes OBJ unter objects/diverse/<br/>MeshConverter mit 5 und mit 10 Hüllen"]
    mc --> ou[("usd/objects/name/name_h5.usd<br/>usd/objects/name/name_h10.usd")]
```

Prüfen: `python -u scripts/check_layout.py` muss mit `PASSED` enden.

---

## 6. Stufe 1: Retargeting trainieren

**Start:** `scripts/train_retarget_smplx.sh --num_envs 2048`
(Umgebungsvariablen `RAW_MOTION_DIR`, `ASSET_SCALE`, `PREPARED_MOTION_DIR` optional.)

**Konfiguration:** `ultra/data/cfg/g1_retarget_smplx.yaml`, `ultra/data/cfg/train/rlg/g1_retarget_smplx.yaml`

```mermaid
flowchart TD
    sh["scripts/train_retarget_smplx.sh"]
    sh --> prep["scripts/prepare_retarget_smplx.py<br/>Clips mit Asset filtern,<br/>Symlinks name_080_080_080.pt"]
    prep --> pd[("InterAct/OMOMO_retarget_supported_080_080_080/")]
    sh --> run["ultra/run.py --task UltraG1 --motion_file ...<br/>--output_path output/retarget_smplx"]
    pd -.-> run
    run --> common["Start wie Kapitel 2 und 3"]

    common --> lm["env/tasks/ultra_g1.py: UltraG1._load_motion"]

    subgraph load ["pro Clip"]
        l1["torch.load, ersten Frame verwerfen"]
        l2["interp_time_series: 30 → 60 Hz"]
        l3["sparseXYZMultiplier auf Root, Keypoints, Objekt"]
        l4["Geschwindigkeiten per finiter Differenz"]
        l5["14 Keypoints über keyIndex"]
        l6["Interaction Graph: compute_sdf zu 256 Objektpunkten"]
        l7["hoi_data (869), hoi_refs (332),<br/>30 Stehframes voran, auf CPU ablegen"]
        l1 --> l2 --> l3 --> l4 --> l5 --> l6 --> l7
    end

    lm --> l1
    l7 --> pad["auf CPU auffüllen, einmal auf die GPU kopieren"]
    pad --> train["learning/ultra_agent.py: UltraAgent.train"]

    subgraph epoch ["pro Epoche: UltraAgent.train_epoch"]
        e1["play_steps: 32 Policy-Schritte (Kapitel 4)<br/>Reset: UltraG1._set_env_state<br/>Standpose, Höhe initRootHeight, feste Gierrichtung"]
        e2["discount_values: GAE"]
        e3["prepare_dataset"]
        e4["6 Mini-Epochen x Minibatches:<br/>train_actor_critic → calc_gradients (PPO)"]
        e5{"epoch % 250 == 0?"}
        e6["self.save → output/retarget_smplx/<br/>g1_retarget_smplx/nn/g1_retarget_smplx.pth"]
        e1 --> e2 --> e3 --> e4 --> e5
        e5 -->|"ja"| e6
    end

    train --> e1
    e5 -->|"nein"| e1
    e6 --> e1
    e4 -.-> rew["Reward: humanoid_g1.py: compute_humanoid_reward<br/>Abbruch: humanoid_g1.py: compute_humanoid_reset"]
```

Während des Trainings: `progress_buf` zählt den Referenzframe; `_compute_observations` liefert 1853 Werte (Referenz t+1 und t+16); Positionsziele gehen an den impliziten PD-Regler von PhysX.

---

## 7. Stufe 2: Export nach `[T, 630]`

**Start:**

```bash
python scripts/export_retarget_smplx.py \
  --input-dir InterAct/OMOMO_retarget \
  --checkpoint output/retarget_smplx/g1_retarget_smplx/nn/g1_retarget_smplx.pth \
  --output-dir output/retarget_export_080 \
  --asset-scale 080_080_080 \
  --xyz 1 1 1 --xyz 1.05 1 0.95
```

```mermaid
flowchart TD
    ex["scripts/export_retarget_smplx.py: main"]
    ex --> flt["unterstützte Clips suchen<br/>(Objekt-URDF und -OBJ vorhanden)"]
    flt --> loop{"für jeden Clip x jede --xyz-Variante"}
    loop -->|"Ziel existiert"| skip["überspringen"]
    loop -->|"neu"| wd["Arbeitsordner OUTPUT_work/clip/scale/xyz:<br/>Symlink auf den Clip"]
    wd --> yml["env.yaml schreiben: numEnvs 1, stateInit Start,<br/>keine Early Termination, rolloutLength 100000,<br/>sparseXYZMultiplier, retargetExportPath<br/>train.yaml: Player 1 Spiel, deterministisch"]
    yml --> sub["subprocess: ultra/run.py --task UltraG1 --test<br/>--checkpoint ... --num_envs 1 --headless"]
    sub --> common["Start wie Kapitel 2 und 3,<br/>Runner.run → Player"]
    common --> prst["UltraPlayerContinuous.restore(checkpoint)<br/>ultra_models.load_checkpoint"]
    prst --> prun["UltraPlayerContinuous.run"]
    prun --> step["pro Schritt: get_action deterministisch → env_step"]
    step --> phs["UltraG1._physics_step"]
    phs --> cap["_capture_retarget_frame<br/>Root, Gelenke, Objekt, 39 Körper, Kontakte von Env 0"]
    cap --> pps["UltraG1.post_physics_step"]
    pps --> done{"reset_buf[0]?<br/>(Ende der Motion)"}
    done -->|"nein"| step
    done -->|"ja"| save["_save_retarget_rollout<br/>30 Stehframes verwerfen, torch.save"]
    save --> out[("OUTPUT/clip_xyz..._080_080_080.pt<br/>[T, 630]")]
    out --> chk["Exporter prüft Rückgabewert und Datei<br/>→ nächste Variante"]
    chk --> loop
```

Pro Clip und Variante startet Isaac Sim einmal neu.

---

## 8. Stufe 3: Teacher trainieren

**Start:** `scripts/train_teacher.sh --num_envs 2048` (Daten: `env.motion_file` in `g1_teacher.yaml`, Standard `InterAct/OMOMO_retarget_aug`, oder `--motion_file <ordner>`).

**Konfiguration:** `ultra/data/cfg/g1_teacher.yaml`, `ultra/data/cfg/train/rlg/g1_teacher.yaml`

Gleiche Kette wie Stufe 1; die Unterschiede liegen in `env/tasks/ultra_g1_retarget.py`:

```mermaid
flowchart TD
    sh["scripts/train_teacher.sh<br/>run.py --task UltraG1Retarget --output_path output/teacher"]
    sh --> common["Start wie Kapitel 2 und 3<br/>Aktuator ohne Gains (Drehmomentregelung),<br/>Objekt mit 10 Hüllen"]
    common --> sep["UltraG1Retarget._setup_env_properties"]
    sep --> sep1["Humanoid_G1: Reibung und Torso-Masse randomisieren (DR)"]
    sep --> sep2["_setup_target_properties<br/>Objektmaterial 0.6 / 0.05,<br/>Masse, Schwerpunkt, Trägheit pro Env"]
    sep1 & sep2 --> lm["UltraG1Retarget._load_motion<br/>[T, 630] lesen, 20 Endframes anhängen,<br/>Interaction Graph für alle 39 Körper"]
    lm --> train["UltraAgent.train (wie Stufe 1)"]

    train --> rst["Reset: _reset_hybrid_state_init<br/>Motion passend zum Objekt, zufälliger Frame,<br/>Stand-still-Envs, Observation-Masken<br/>_set_env_state: Referenzpose + Rauschen"]
    rst --> phys["_physics_step: _compute_torques in jedem der 17 Physikschritte,<br/>Motorstärke-Randomisierung"]
    phys --> post["post_physics_step: Stösse, Schwerkraft (DR)"]
    post --> obs["_compute_observations: 4052 Werte"]
    obs --> rew["_compute_reward: compute_humanoid_reward · 1.6<br/>+ SmoothRewards (additiv, Curriculum)"]
    rew --> res["_compute_reset: compute_humanoid_reset"]
    res --> ck[("output/teacher/g1_teacher/nn/g1_teacher.pth<br/>alle 250 Epochen")]
```

---

## 9. Stufe 4: Student destillieren

**Start:** `scripts/train_student.sh --num_envs 2048`, mehrere GPUs: `NUM_GPUS=4 scripts/train_student_multigpu.sh` (torchrun, `--multi_gpu`).

**Konfiguration:** `ultra/data/cfg/g1_student_vae.yaml` (`env.teacherPolicy`), `ultra/data/cfg/train/rlg/g1_student_vae.yaml`

```mermaid
flowchart TD
    sh["scripts/train_student.sh<br/>run_distill.py → run.py --task UltraDistillObjV2Point"]
    sh --> common["Start wie Kapitel 2 und 3"]
    common --> init["env/tasks/ultra_g1_distill_obj_v2vae.py:<br/>UltraDistillObjV2Point.__init__<br/>(erbt von UltraG1Retarget)"]
    init --> tp["learning/ultra_models.py: load_teacher_policy<br/>Teacher-Netz + Normalisierung, eingefroren"]
    tp --> wrap["isaac/vec_task.py: UltraDAggerVecTask<br/>gibt Student-Obs + Teacher-Ausgabe zurück"]
    wrap --> train["learning/ultra_agent_distill_vae.py:<br/>UltraAgentDistill.train"]

    subgraph ps ["UltraAgentDistill.play_steps"]
        k0["update_obs_keep_probabilities(epoch)<br/>Modalitäten seltener sichtbar"]
        k1["beta_t: Anteil Teacher-Aktionen,<br/>1 bis Epoche 50, dann bis 0 bei Epoche 550"]
        k2["env_reset → (Student-Obs, Teacher-Ausgabe)"]
        k3["get_action_values: Student-Netz (VAE),<br/>mit Wahrscheinlichkeit beta_t Teacher-Aktion ausführen"]
        k4["env_step"]
        k0 --> k1 --> k2 --> k3 --> k4
        k4 --> k2
    end

    train --> k0

    subgraph envstep ["UltraDistillObjV2Point.step"]
        s1["pre_physics_step, _physics_step (Drehmoment)"]
        s2["post_physics_step<br/>+ _compute_observations_student (1496):<br/>Propriozeption, Punktwolke Kopfkamera, Ziele, Masken"]
        s3["Teacher-Obs normalisieren, _mask_teacher_obs,<br/>Teacher-Netz → action_buf, mu_buf"]
        s1 --> s2 --> s3
    end

    k4 --> s1
    s3 --> k2

    k4 --> cg["calc_gradients:<br/>(μ_student − μ_teacher)² + VAE-KL + Glättung + Hilfsverluste<br/>PPO-Terme mit Faktor 0"]
    cg --> ck[("output/student/g1_student_vae/nn/g1_student_vae.pth<br/>alle save_frequency Epochen")]
```

---

## 10. Stufe 5: Student finetunen

**Start:** `scripts/train_finetune.sh <student.pth> --num_envs 2048`

**Konfiguration:** `ultra/data/cfg/g1_student_finetune.yaml`, `ultra/data/cfg/train/rlg/g1_student_finetune.yaml`

```mermaid
flowchart TD
    sh["scripts/train_finetune.sh student.pth<br/>run.py --task UltraDistillObjV3RL --resume_from student.pth"]
    sh --> common["Start wie Kapitel 2 und 3"]
    common --> env["env/tasks/ultra_g1_distill_obj_v3rl.py:<br/>UltraDistillObjV3RL (erbt von UltraG1Retarget)<br/>eigener step, reset, _compute_reward"]
    env --> ag["learning/ultra_agent_distill_vae_rl.py:<br/>UltraAgentDistill.train"]
    ag --> rest["UltraAgent.train: restore(resume_from)<br/>Student-Gewichte laden"]
    rest --> ps["play_steps: wie Stufe 4, mit Teacher-Ausgabe"]
    ps --> cg["calc_gradients:<br/>Destillationsverlust (über distill_mask)<br/>+ PPO-Actor/Critic-Verlust (über prior_mask)"]
    cg --> ck[("output/finetune/g1_student_finetune/nn/g1_student_finetune.pth")]
    ck --> ps
```

---

## 11. Abspielen und Inferenz

```mermaid
flowchart TD
    subgraph cmds ["Startbefehle"]
        pt["scripts/play_teacher.sh ckpt.pth [num_envs]<br/>run.py --task UltraG1Retarget --test"]
        psd["scripts/play_student.sh ckpt.pth [num_envs] [task_mode] [obj_obs]<br/>run_distill.py --task UltraDistillObjV2Point --test"]
        pds["scripts/play_dataset.sh<br/>run_distill.py ... --test --play_dataset"]
        ti["python ultra/run_teacher_inference.py --motion-dir ordner_mit_einem_clip"]
        r1["run.py ... --task UltraG1 --test --checkpoint retarget.pth<br/>(Retargeting-Policy anschauen)"]
    end

    pt & r1 --> pl["UltraPlayerContinuous.run"]
    psd --> pld["UltraPlayerContinuousDistill.run"]
    pds --> pld

    pl --> pq{"play_dataset?"}
    pld --> pq
    pq -->|"nein"| inf["Schleife: env_reset(done) → get_action → env_step → render"]
    pq -->|"ja"| dat["Schleife über Frames: task.play_dataset_step(t)<br/>Referenzzustand direkt schreiben"]

    ti --> tir["ersetzt UltraPlayerContinuous.run durch run_rollout,<br/>startet run.py mit g1_teacher_no_dr.yaml"]
    tir --> tio[("output/teacher_inference/rollout.pt<br/>Zustand, Referenz, Aktionen")]
```

Mit Fenster: `--headless` weglassen und wenige Envs wählen (Rendern verlangsamt stark). MuJoCo statt Isaac Lab: `scripts/sim2sim_teacher.sh`, `scripts/sim2sim_student.sh`; TorchScript-Export für das Deployment: `scripts/export_jit.sh`.
