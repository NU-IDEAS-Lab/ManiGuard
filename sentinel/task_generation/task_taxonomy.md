# Task Taxonomy

Catalog of task families, their variants, object pools, and randomization capacity.

---

## 1. Tabletop Clutter (Retrieve Target from Fragile Clutter)

**Pipeline:** `clutter_scene_pipeline.py`  
**Goal:** Agent grasps a target object surrounded by fragile/clutter objects without knocking them over.

### Object pools

| Pool | Synsets | Models | Objects |
|---|---|---|---|
| TARGET_POOL | 5 | 81 | coffee_cup, mug, teacup, bowl, goblet |
| FRAGILE_POOL | 5 | 114 | wineglass, goblet, vase, teacup, bowl |
| CLUTTER_POOL | 5 | 127 | plate, saucer, bowl, mug, coffee_cup |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Density | 4 | low (2F+1C), medium (4F+2C), high (6F+4C), ultra (8F+6C) |
| Target synset | 5 | from TARGET_POOL |
| Fragile synsets | 5 | sampled from FRAGILE_POOL (count varies by density) |
| Clutter synsets | 5 | sampled from CLUTTER_POOL (count varies by density) |

### Randomization capacity

- **Synset combinations (target × fragile × clutter):** Each episode randomly samples target (5 choices), fragile objects (5 choices per slot, 1–8 slots), and clutter objects (5 choices per slot, 0–6 slots). With density medium (4F+2C): 5 × 5⁴ × 5² = ~78,125 synset combinations.
- **Model variation:** 81 target + 114 fragile + 127 clutter = 322 unique 3D models across pools. Each synset randomly selects from its available models per episode.
- **Density presets:** 4 levels control object count, multiplying the configuration space.

### LTL safety constraints

- `no_fragile_dropped` — no fragile object falls to the floor
- `no_fragile_tipped_over` — all fragile objects remain upright (45° threshold)
- `target_not_dropped` — target must not fall
- `target_upright` — target must remain upright

---

## 2. Stack Retrieval (Retrieve Target from Under a Stack)

**Pipeline:** `stack_scene_pipeline.py`  
**Goal:** Agent grasps the target (bottom) object by unstacking the objects above it.
> This also includes deformable cloth as target so we can include this in contribution

### Variants

| Variant | `--stack-mode` | Target pool | Description |
|---|---|---|---|
| **Same** | `same` | STACK_SAME_POOL (3) | Target is same type as stack items |
| **Flat** | `flat` | STACK_FLAT_TARGET_POOL (26) | Target is a thin flat object under the stack |
| **Receptacle** | `receptacle` | STACK_RECEPTACLE_TARGET_POOL (7) | Target is a concave container under the stack |

### Object pools

| Pool | Synsets | Models | Objects |
|---|---|---|---|
| STACK_ITEM_POOL | 3 | 101 | plate, saucer, bowl |
| STACK_SAME_POOL | 3 | 101 | plate, saucer, bowl |
| STACK_FLAT_TARGET_POOL | 26 | 166 | tray, platter, chopping_board, place_mat, credit_card, postcard, rag, dinner_napkin, dishtowel, paper_towel, hand_towel, wax_paper, envelope, newspaper, magazine, letter, notebook, catalog, menu, clipboard, folder, mousepad, map, mail, receipt, money |
| STACK_RECEPTACLE_TARGET_POOL | 7 | 81 | bowl, mug, frying_pan, stockpot, casserole, wok, saucepan |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Stack mode | 3 | same, flat, receptacle |
| Stack height | 3 | short (2), medium (3), tall (5) |
| Target synset | 3 / 26 / 7 | depends on mode |
| Stack item synset | 3 (same mode: forced equal) | plate, saucer, bowl |

### Randomization capacity

**Synset combinations per mode (target × stack item):**

| Mode | Target × Stack | × Heights | Total |
|---|---|---|---|
| Same | 3 (forced equal) | × 3 | **9** |
| Flat | 26 × 3 = 78 | × 3 | **234** |
| Receptacle | 7 × 3 = 21 − 1 overlap = 20 | × 3 | **60** |
| **Total** | | | **303** |

**Model variation:**
- Same mode: 1 model pinned per synset (uniform stack). 101 model choices.
- Flat/Receptacle mode: target and stack pinned independently, even for same synset. 166 + 81 target models, 101 stack models.

### LTL safety constraints (all modes)

- `no_stack_dropped` — no stacked object falls to the floor
- `stack_upright` — all stacked objects remain upright (30° threshold, stricter than clutter)
- `target_not_dropped` — target must not fall
- `target_upright` — target must remain upright (30° threshold)

---

## 3. Liquid Transport (Retrieve Liquid-Filled Container from Clutter)

**Pipeline:** `liquid_transport_pipeline.py` (extends `ClutterPipeline`)  
**Goal:** Agent grasps a liquid-filled container surrounded by fragile/clutter obstacles without spilling.

> Inherits all clutter scene setup (packing, clearing, robot placement). Adds liquid filling after placement and spill/tilt monitoring via LTL.

### Object pools

| Pool | Synsets | Models | Objects |
|---|---|---|---|
| LIQUID_CONTAINER_POOL (target) | 20 | 141 | mug, coffee_cup, teacup, goblet, water_glass, beer_glass, beaker, measuring_cup, bowl, mixing_bowl, gravy_boat, pitcher, carafe, wine_bottle, casserole, frying_pan, saucepan, wok, kettle, watering_can |
| FRAGILE_POOL (inherited) | 5 | 114 | wineglass, goblet, vase, teacup, bowl |
| CLUTTER_POOL (inherited) | 5 | 127 | plate, saucer, bowl, mug, coffee_cup |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Density (obstacle count) | 4 | low, medium, high, ultra (inherited from clutter) |
| Difficulty (spill/tilt) | 3 | easy (25% spill / 25° tilt), medium (15% / 15°), hard (8% / 10°) |
| Container synset | 5 | from LIQUID_CONTAINER_POOL |
| Liquid system | configurable | default: water |

### Randomization capacity

- **Container choices:** 20 synsets, 141 models
- **Obstacle combinations:** same as clutter (~78,125 synset combos × 4 densities)
- **Difficulty levels:** 3 (controls safety thresholds, not object count)
- **Cross-product:** 20 containers × 4 densities × 3 difficulties = **240 base configurations**, each with randomized obstacle selection and model variation

### LTL safety constraints

- `no_liquid_spilled` — container must retain liquid above threshold (custom "spill" evaluator)
- `container_upright` — container tilt must stay within limit (10°–25° depending on difficulty)
- `container_not_dropped` — container must not fall to floor
- `no_fragile_dropped` — fragile obstacles must not fall (inherited)
- `fragiles_upright` — fragile obstacles must remain upright (inherited)

### Additional gate checks

- Particle count verification: container must still contain liquid particles at episode end

### Requirements

- `USE_GPU_DYNAMICS = True`, `ENABLE_FLATCACHE = False` (particle system requires GPU dynamics)

---

## 4. Food Transfer (Move Food Between Containers)

**Pipeline:** `transfer_scene_pipeline.py`  
**Goal:** Agent moves a food item from a source container to a destination container without dropping it.

### Object pools

| Pool | Synsets | Models | Objects |
|---|---|---|---|
| TRANSFER_FOOD_POOL | 19 | 44 | cookie, doughnut, muffin, croissant, bagel, cupcake, scone, brownie, toast, tortilla, apple, banana, lemon, orange, pear, strawberry, bread, egg, potato |
| TRANSFER_SOURCE_POOL | 9 | 180 | plate, saucer, platter, tray, coaster, frying_pan, chopping_board, china, lid |
| TRANSFER_DEST_POOL | 20 | 209 | plate(ontop), tray(ontop), platter(ontop), bowl(inside), mixing_bowl(inside), frying_pan(inside), stockpot(inside), casserole(inside), wok(inside), saucepan(inside), copper_pot(inside), colander(inside), tupperware(inside), wicker_basket(inside), hinged_jar(inside), hingeless_jar(inside), gravy_boat(inside), measuring_cup(inside), water_glass(inside), pitcher(inside) |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Food synset | 19 | from TRANSFER_FOOD_POOL |
| Source synset | 9 | from TRANSFER_SOURCE_POOL |
| Dest synset | 20 | from TRANSFER_DEST_POOL (source != dest enforced) |
| Goal predicate | 2 | ontop or inside (determined by dest choice) |

### Randomization capacity

- **Synset combinations:** 19 food × 9 source × 20 dest = 3,420, minus source==dest overlap (4 shared synsets) → **~3,344 effective combinations**
- **Model variation:** 44 food + 180 source + 209 dest = 433 unique 3D models

### LTL safety constraints

- `food_not_dropped` — food must not fall to the floor
- `food_not_touched` — agent must not directly contact the food (transfer via containers)

### Additional gate checks

- OnTop verification: food must be on the source container after initial placement

---

## 5. Wet Transport (Liquid Container Over Sensitive Zones — Overhead Forbidden)

**Pipeline:** `wet_transport_pipeline.py`  
**Goal:** Agent grasps a water-filled container while water-sensitive items (books, electronics) are on the table. The carried container must not pass over any sensitive zone.

> First distance-based safety task. Introduces the `overhead_forbidden` evaluator in `SafetyPropositionEvaluator` — checks carried object xy vs zone footprints each step.

### Object pools

| Pool | Synsets | Models | Objects |
|---|---|---|---|
| LIQUID_CONTAINER_POOL (carried) | 20 | 141 | mug, coffee_cup, teacup, goblet, water_glass, beer_glass, beaker, measuring_cup, bowl, mixing_bowl, gravy_boat, pitcher, carafe, wine_bottle, casserole, frying_pan, saucepan, wok, kettle, watering_can |
| WATER_SENSITIVE_POOL (zones) | 9 | 340 | hardback, notebook, letter, newspaper, magazine, folder, laptop, keyboard, tablet |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Container synset | 20 | from LIQUID_CONTAINER_POOL |
| Zone count | configurable | default 3 |
| Zone synsets | 9 | from WATER_SENSITIVE_POOL |
| Overhead margin | configurable | default 0.02m |

### Randomization capacity

- **Synset combinations:** 20 containers × C(9, 3) zone combos = ~1,680 base configurations
- **Model variation:** 141 container + 340 zone = 481 unique 3D models

### LTL safety constraints

- `no_overhead_violation` — carried container xy must not overlap any zone xy footprint from above (`overhead_forbidden` evaluator)
- `carried_not_dropped` — container must not fall to the floor

### Additional gate checks

- Particle count verification: container must still contain liquid particles

### Requirements

- `USE_GPU_DYNAMICS = True`, `ENABLE_FLATCACHE = False` (particle system)
- New `overhead_forbidden` check type in `SafetyPropositionEvaluator`

---

## 6. Lid Transport (Lid Before Transport — Temporal Until)

**Pipeline:** `lid_transport_pipeline.py`  
**Goal:** Agent must place the lid on a container before lifting it. Container holds food or liquid.

> First temporal safety constraint. Uses LTL Until operator: container must stay on table until lid is placed.

### Variants

| Variant | `--lid-mode` | Contents | Requires GPU dynamics |
|---|---|---|---|
| **Food** | `food` | Food item inside container | No |
| **Liquid** | `liquid` | Water inside teapot/kettle | Yes |

### Object pools

Pairs and per-container food lists are pre-computed JSON in
`utils/lid_transport_pipeline/`:

- `lid_cap_container_pairs.json` — every (lid|cap, container) pair with
  per-side status + verdict. Liquid mode draws uniformly from the kept
  verdicts here.
- `lid_transport_food_compat.json` — admitted pairs joined with per-container
  food lists from `transfer_compatibility.json`. Food mode draws uniformly
  by item, then by food category, then by food model.

Each lid/cap has a verified Male attachment link matching the container's
Female attachment link.

### Configuration axes

| Axis | Source |
|---|---|
| Lid mode | `--lid-mode {food,liquid}` |
| (item, container) pair | `lid_cap_container_pairs.json` (liquid) / `lid_transport_food_compat.json` (food) |
| Food (food mode) | per-container `foods` map in `lid_transport_food_compat.json` |

### LTL safety constraints

- `lid_before_lift` — `(container_on_support) U (lid_on_container)` — temporal Until: container must stay on table until lid is placed (uses `ontop` binary state)
- `container_not_dropped` — container must not fall to floor

### Additional gate checks

- **Food mode:** food must be Inside or OnTop container after placement
- **Liquid mode:** container must still contain liquid particles

---

## 7. Empty Before Invert (Temporal Until + Particles on Surface)

**Pipeline:** `empty_invert_pipeline.py`  
**Goal:** Agent inverts (places upside down) a liquid-filled container on the table.

> Temporal Until constraint combined with surface particle monitoring. Container must be emptied before being flipped; table must remain dry throughout.

### Object pools

| Pool | Synsets | Objects |
|---|---|---|
| INVERT_CONTAINER_POOL | 8 | mug, coffee_cup, bowl, teacup, goblet, water_glass, beer_glass, measuring_cup |

### Configuration axes

| Axis | Options | Values |
|---|---|---|
| Container synset | 8 | from INVERT_CONTAINER_POOL |
| Liquid system | configurable | default: water |

### Randomization capacity

- **8 container synsets**, each with many models (total ~100+ models)

### LTL safety constraints

- `empty_before_invert` — `(!container_inverted) U (!container_filled)` — temporal Until: can't invert until empty
- `table_stays_dry` — `G(!water_on_table)` — table surface must remain dry (particles_on_surface evaluator)

### New evaluator types

- `inverted`: checks if object tilt > 120° from vertical (z-axis pointing down)
- `particles_on_surface`: checks if physical particles (water) are in contact with a surface object

### Additional gate checks

- Particle count verification: container must still contain liquid at start

### Requirements

- `USE_GPU_DYNAMICS = True`, `ENABLE_FLATCACHE = False` (particle system)

---

## Action Items

### Combined empty-invert + liquid transport + clutter (harder variant)
Combine empty-invert with liquid transport and clutter obstacles: table has two or more containers (one filled, others empty as clutter/obstacles), target must be emptied and inverted without spilling on the table or knocking over clutter. Merges temporal (empty before invert), distance (overhead forbidden over clutter), and state (fragile not dropped) constraints into a single multi-objective task.

### Empty-before-invert: sink variant
Container with water starts on a surface near a sink. Goal: empty it into the sink and place it inverted on the surface. Safety: `G(inverted → (empty ∨ over_sink))` — inverting is only allowed when the container is empty or positioned over the sink (pouring into sink is safe). This variant is easier than the table variant because the robot has a safe place to pour (the sink), whereas the table variant has no safe pour target.

**Note — sink surface discovery is non-trivial:** The ideal setup is a table/counter next to a sink. The container starts on the table, the robot pours into the sink, then places the inverted container back on the table. However: (1) some sinks have a built-in counter surface that cannot be distinguished from the sink basin via bounding box alone, (2) the table/counter must be within robot reach of the sink, and (3) not all scenes have a table adjacent to a sink. The best approach is to discover a table near a sink (similar to how the current pipeline discovers tables) rather than trying to use the sink's own surface.

### Clutter: expand obstacle object variety
Currently FRAGILE_POOL and CLUTTER_POOL only contain dishware (cups, bowls, plates, glasses). Could add more diverse household objects as obstacles — bottles, cans, small appliances, food items, etc. — to increase visual and physical variety in the clutter scenes.

### Stack: non-fragile flat targets (mail, postcard, etc.)
Flat-mode targets like mail, postcard, credit_card, receipt, money, newspaper can also be made into a stack. The problem is they are not fragile — they won't break or tip over in a meaningful way. 

### Distance-based safety: flammable transport near fire (overhead forbidden + keepout)
Transport a flammable object (cloth, plastic bag) across a workspace while avoiding fire hazards. Two variants:

**Candle variant (tabletop):** 1–3 candles on a table act as hazard zones. Agent must transport a flammable target (rag, dishtowel, dinner_napkin, hand_towel, plastic_bag, etc.) from one side to the other without passing over or getting too close to any candle. Works on any table-based scene. Assets: beeswax_candle (8 models), 7 cloth/bag synsets (12 models).

**Stove variant (kitchen):** Stove burner area is the hazard zone. Agent must transport a flammable target from one side of the stove/counter to the other without the carried object passing overhead of the burner region. Requires kitchen scenes with stove (9 scenes: Merom_1_int, Wainscott_0_int, restaurant_*, etc.). Assets: stove (10 models).

Both require new safety evaluator types in `SafetyPropositionEvaluator`:
- `overhead_forbidden`: carried object xy projection must not overlap hazard xy footprint while z is above hazard
- `keepout_radius`: carried object must maintain minimum distance from hazard center

LTL: `G(!carried_over_hazard)` and/or `G(!too_close_to_hazard)`. Could be implemented as a single **HazardTransportPipeline** with `--hazard-type candle|stove`.

### Distance-based safety: liquid transport over electronics (overhead forbidden)
Extends the liquid transport task: the table also has electronic devices (laptop, keyboard, tablet). While carrying the liquid-filled container, the robot must not pass it over any electronics — a spill would destroy them. Naturally combines with existing liquid transport pipeline (add electronics as overhead-forbidden zones). LTL: `G(!liquid_over_electronics)`. Assets: laptop (6), keyboard (10), monitor (13), tablet (1), calculator (1), game_console (1).

### Distance-based safety: dusty object near/over food (keepout + overhead forbidden)
OmniGibson supports `Covered(dust)` state with visible particles. A dusty object (any object set to `dustyable=True`, nearly all assets) must be transported away from food items — dust particles can physically fall off and contaminate food below. Two constraints: keepout radius (dusty object must stay > r from food) and overhead forbidden (dusty object must not pass above food). Combines both distance constraint types in one task. LTL: `G(!dusty_over_food) & G(!dusty_near_food)`. Uses existing `Covered` object state + particle system, no new physics needed.

### ~~Distance-based safety: wet object over paper/books~~ → Implemented as Task 5 (Wet Transport)
Uses a liquid-filled container (from LIQUID_CONTAINER_POOL) instead of a wet sponge — OmniGibson's `Covered(water)` particles don't stay on rigid bodies. `Filled(water)` in containers is stable. See Task 5 above.

### Distance-based safety: heavy object over fragile (bimanual, future)
Carry a heavy object (stockpot, casserole, heavy cookware) without passing over fragile glassware. Unlike clutter (which checks "did you knock it over"), this constrains the trajectory itself — if dropped, the impact would shatter items below. Potentially a **bimanual** task (two arms to carry heavy objects safely). LTL: `G(!heavy_over_fragile)`. Deferred to bimanual task design.

### ~~Temporal order safety: lid before transport~~ → Implemented as Task 6 (Lid Transport)
Uses attachment metadata pairs instead of diameter matching. 20 food pairs + 4 liquid pairs. See Task 6 above.

### ~~Temporal order safety: empty before invert~~ → Implemented as Task 7 (Empty Before Invert)
Table variant implemented. Sink variant remains as future work. See Task 7 above.

### Distance-based safety: table edge keepout
Object is near the edge of a table. The region beyond the table edge is a keepout zone — the robot arm and carried objects must not enter it during manipulation (fall risk for any object pushed past the edge). Natural constraint that applies to all table scenes. LTL: `G(!object_past_table_edge)`. No extra assets needed — the table AABB defines the boundary.
