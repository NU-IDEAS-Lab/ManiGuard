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

## Action Items

### Clutter: expand obstacle object variety
Currently FRAGILE_POOL and CLUTTER_POOL only contain dishware (cups, bowls, plates, glasses). Could add more diverse household objects as obstacles — bottles, cans, small appliances, food items, etc. — to increase visual and physical variety in the clutter scenes.

### Stack: non-fragile flat targets (mail, postcard, etc.)
Flat-mode targets like mail, postcard, credit_card, receipt, money, newspaper can also be made into a stack. The problem is they are not fragile — they won't break or tip over in a meaningful way. 
