# Benchmark run common bugs summary

## 01_stack_same_1200s_20260407

### Common issues

- Infra / server-side GPU OOM
这是 stdout-only failed runs 的主因，不是单纯 timeout。典型签名是：
Skipping NVIDIA GPU due CUDA being in bad state -> ERROR_OUT_OF_DEVICE_MEMORY -> Fatal Python error:
Segmentation fault
- RuntimeError: No suitable table-like surface found in scene
说明 scene 本身没找到合格 tabletop support，不是 server crash。
- Camera contract problems
    - main cam 被挡住 / 没把 Franka + tabletop object 一起框进来
    - wrist cam 因为 EEF pose / wrist pose 不对，直接看天花板或偏离桌面
    - 需要对齐 zero-shot inference 里更 canonical 的 main cam + wrist cam pose logic
- Support region / placement geometry problems
    - 一些桌子不是标准 rectangular tabletop，AABB / selected rect 可能落在中空区域，导致 object initial
    placement 悬空，sim 一开始就 free-fall
    - L-shape / hollow-center support 特别容易出这个问题
    - 有些 scene 里 robot placement 落在 rectangle 的“短边侧”，reachability 不如“长边侧”；这个最好结合 support
    profile region 的 long/short axis 来决定 Franka 放置侧
- Stack stability problems
    - stack 太高，或者 stacked objects 初始对齐不够规整，sim 一开始就倒，后续直接触发 ltl_violated

### Representative scene notes

- Beechwood_1_int
没有 video；log 明确更像 server-side GPU OOM crash，不是 scene logic failure
- Benevolence_1_int
同上，log 看起来也是 GPU OOM / segfault
- Benevolence_2_int
third-person main cam 被挡，需要参考 canonical zero-shot main cam pose
- grocery_store_asian
main cam 被挡；wrist cam 看天花板，EEF / wrist pose 明显不对
status=success, gate_pass=False, ltl_violated=True
- grocery_store_cafe
桌面上没东西；高度怀疑是 support 不是标准长方形，导致 object 悬空放置后直接自由落体
status=success, gate_pass=False, ltl_violated=True
- grocery_store_half_stocked
main cam 没找到 Franka + tabletop objects；Franka 在画面里看起来也有悬空嫌疑，需 double check
同时这个 support 看起来像 L-shape / hollow-center，AABB center placement 很可能导致 free-fall
status=success, gate_pass=False, ltl_violated=True
- hall_conference_large
这个是明确 timeout，status=timeout, duration_s=1203.4；没有看到决定性 runtime error
- hall_glass_ceiling
third-person cam angle 有问题
- hotel_gym_spa
选到了健身躺椅一类 support，不像应保留的 tabletop candidate；建议考虑 exclude
- hotel_suite_small
main cam 被挡
- house_double_floor_lower
不是 OOM；是明确 No suitable table-like surface found in scene 后 shutdown
- office_cubicles_left
bowl/stack 太高且不够规整，对齐差，初始就摔
status=success, gate_pass=False, ltl_violated=True
- Pomaria_2_int
同类问题：stack 太高 / 不够对称，容易初始 collapse
status=success, gate_pass=True, ltl_violated=True
- office_cubicles_right
很可疑：surface profile 里像规整 conference table，但 video 看起来像桌子 upside-down / 把桌脚当桌面；也要怀
疑 cleanup 是否误处理了 tabletop geometry
status=success, gate_pass=True, ltl_violated=True

## 02_liquid_transport_20260407

### Common issues

- Camera / view contract problems
    - main cam framing 不稳定，有些 scene 没有稳定看到 Franka + tabletop objects
    - 个别 scene 甚至出现 main cam 对天 / object 从画面高处坠落的异常观感
- Support geometry / placement region problems
    - 桌子 shape 不规则，AABB / selected region 误判，导致 object 没有形成合理有效的摆放
    - footprint 太大时，objects 被挤到 edge，接近掉落，甚至初始就已有 object 掉地
    - 某些 scene 看起来 object count 异常偏少，怀疑是 pack/cull 过头，或者 support budget 判断过于保守
- Initial placement / collision / drift problems
    - object 与桌面之间可能有 small gap，sim 一开始会“原地晃一下”
    - 某些 scene table surface 看起来平，但 object 会持续 slow drift，怀疑 initial placement already has
    collision / contact instability
    - liquid container 太大时，会和 surrounding objects 甚至 Franka init pose 互相挤占
- Only-log / incomplete-run problems
    - 这组和 01_stack_same 不同，不是以 server GPU OOM 为主
    - 主要是：
        - pack_generation_failed_after_retries
        - No suitable table-like surface found
        - 某些 scene asset transform / emitter assertion -> segfault
        - 以及若干真实 1200s timeout

### Only-log scenes: log-confirmed causes

- Packing failure
    - Benevolence_2_int
    pack_generation_failed_after_retries: zone_capacity_exceeded
    含义：clutter/liquid target pack 不进 red zone，support region / usable area 不够
    - grocery_store_half_stocked
    pack_generation_failed_after_retries: support_state_failed:frying_pan.n.01_1
    含义：不是 simple capacity 问题，而是 support/target state setup 本身反复失败
- No valid tabletop support
    - Wainscott_0_garden
    RuntimeError: No suitable table-like surface found in scene
- Scene asset transform / emitter bug -> segfault
    - Pomaria_0_garden
    - Pomaria_0_int
    - Pomaria_2_int
    共同签名：AssertionError: ... armchair_* / emitter local transform is not orthogonal -> segfault
    更像 scene asset issue，不是 liquid task logic 本身
- Early Isaac / viewport startup segfault
    - Rs_int
    没有明确 task-level exception，更像 runtime startup crash
- True timeout
    - hall_conference_large
    - house_double_floor_upper
    - school_biology
    - school_chemistry
    - school_computer_lab_and_infirmary
    - school_geography
    这些是真的跑满 1200s，大多卡在 initialization / kinematic sampling 阶段，不像 hard crash

### Representative reviewed scenes

- grocery_store_asian
桌子 shape 导致 bbx / region 判断不准，placement 不合理
- hotel_suite_large
object footprint 太大，摆放已经逼近 edge，甚至已有 object 掉地
- hotel_suite_small
桌面 objects 明显偏少，怀疑被 over-cull / over-prune，需 double check 是否是 generator bug
- house_double_floor_lower
initial placement 可能与桌面之间有 gap，sim 开始时 object 会“原地晃一下”
- house_single_floor
table profile 看起来是平整水平的，但 bowl 仍 low-speed drift，怀疑 init placement collision/contact 不干净
- Merom_0_garden
placement 出大问题：main cam 朝天，object 甚至像是从天空坠落；ltl_violated=True
- Merom_0_int
object 数量看起来过少，和 table capacity 不匹配；gate_pass=False
- office_cubicles_left
placement 太松散，周围 objects 有掉桌风险
- restaurant_cafeteria
桌上只有一个杯子，objects 太少；Franka 似乎还和 nearby chairs 有 collision
- restaurant_diner
liquid container 太大，挤占了其他 objects 的 placement 空间，甚至可能和 Franka init state collision
- restaurant_urban
不是单纯 “video file 坏了”，而是 run 本身不完整：status=timeout, exceeded 1200s；目录里只有
rollout_opposite_side_front_ep1.mp4 + scene_ep1.json + stdout.log，缺 diagnostics.jsonl 和主 rollout_ep1.mp4

## 03_stack_receptacle_1200s_20260408

### Only-log scenes

- hotel_gym_spa
- office_cubicles_left

结论
这 2 个 only-log task 都不是单纯 timeout，也不是典型 GPU OOM，而是 task-generation / scene semantics failure。

- outputs/benchmark_runs-other_tasks/03_stack_receptacle_1200s_20260408/hotel_gym_spa
  - `RuntimeError: No suitable table-like surface found in scene`
  - root cause 是 support discovery failure / no valid tabletop support
  - 不是 server crash，后续中断只是连锁反应

- outputs/benchmark_runs-other_tasks/03_stack_receptacle_1200s_20260408/office_cubicles_left
  - pre-sampling 已经报 room/object sampling mismatch：`Room type [meeting_room] ... cannot sample all the objects needed`
  - 随后报 `RuntimeError: stack expects a non-empty TensorList`，最后 segfault
  - root cause 是 failed sampling 导致 empty task state，不是 server OOM

这组 only-log 的共性

- 都属于 task-generation / scene semantics 问题
- 不是 1200s timeout
- 也不是像前面几组那样以 `ERROR_OUT_OF_DEVICE_MEMORY` 为主
- 可归成两类：
  - no valid tabletop support
  - object/room sampling mismatch -> empty task state

### Common issues

- 任务设计逻辑问题：见下方 `Task design logic bugs/problems`，这里不重复展开
- `inside` / `ontop` 关系定义和实际几何不匹配，很多 receptacle scene 没做到真的 “放进去”，而是变成 `stack ontop`、半卡住、或直接 interpenetration
  - `Beechwood_0_int`：ontop / inside 关系导致 collision 乱飞
  - `house_double_floor_upper`：container 口径太小，实际变成 `stack ontop not inside`
  - `house_single_floor`：inside / ontop 关系不稳定，疑似 collision 冲突后爆炸乱飞
  - `restaurant_brunch`：很多 receptacle target 口径太小，上方物体只是放在上面，不是真的 inside
- 尺寸 / stability mismatch，导致初始 stack 本身就不稳
  - `Benevolence_1_int`：放进去/摞起来的物体并没有足够小且稳定，simulation 开始后仍持续晃动甚至摔倒
- 相机视角问题
- 桌子 shape / hollow support 问题，导致物体悬空摆放或初始化后自由落体
- reachability 问题
  - `office_vendor_machine`：桌子可能太大，Franka 够不着；后续应结合 surface profile 进一步限制可达工作区

### Task design logic bugs/problems

#### 第一，upright 对 wok / pan 这种带把手容器很危险

- 当前 `target_upright` / `stack_upright` 的判定，是用物体 local `+Z` 和 world `+Z` 的夹角做阈值判断；stack
task 这里阈值还是 30 deg。
- 这不是 stable resting pose aware 的定义，不理解“锅靠把手自然微微后仰但仍稳定静止”这种情况。
- 对 `wok` / `frying_pan` / `saucepan` 这类带把手 target，如果它们天然静止姿态就超过阈值，那么 agent 后续可能
必须一直扶着 target 才能不 violate LTL。
- 这会把 benchmark 难点错误地从 retrieval / manipulation planning，变成 object-frame-dependent 的姿态定义问
题。
- 所以这里不是 policy 的问题，而是 `upright` definition 可能不适用于 handled cookware / concave receptacle
target。

#### 第二，task intent 和实际编码目标不一致

- 这组 `stack_receptacle` 当前实际编码的不是 “clear container contents / clear upper objects and then retrieve
container”。
- 它本质上仍然是 `OnTop` vertical stack retrieval：只是 bottom target 换成了 receptacle 类物体。
- BDDL goal 只有 `(grasped agent target)`，没有任何 goal 要求：
- 先把上方/相关物体移开
- 把移开的物体稳定放到桌面别处
- grasp target 前 target 必须已经 unobstructed / cleared
- 因此设计预期中的 “先清理再拿 target” 没有被显式 encode 到 task semantics 里。

#### 第三，当前 spec 允许 policy 走 direct-grasp shortcut

- 对 policy 来说，这个任务很容易被抽象成一个 `safe grasp target` 问题，而不是一个需要显式 reasoning about
container/stack contents 的问题。
- 只要存在一个直接 grasp target 的 pose，并且动作过程中没有触发 drop / tip violation，这条 shortcut 在当前
spec 下就是合法解。
- 换句话说，receptacle 上/里的其他物体，在当前设定里更多只是 physical obstacle，而不是必须被处理的任务语义对
象。
- 所以 benchmark 目前没有真正强制出 `unstack / clear-then-retrieve` 这类中间计划。
- 这里还要注意，`gate_pass` 是 task generation / curation 阶段的初始 scene sanity screen，不是 policy rollout
时去完成的东西；scene 一旦先天 `gate_pass=True`，后续 policy 仍然可能用 shortcut 过关。

#### 第四，`grasped` 的底层语义和当前 Franka runtime 没有完全对齐

- OG 里的 BDDL `grasped(agent, obj)` 最终依赖 robot 内部的 `_ag_obj_in_hand` bookkeeping。
- `_ag_obj_in_hand` 表示的是 “系统登记这个 object 当前在手里”，不是 raw eef contact，也不是严格的 stable
physical grasp / force-closure 定义。
- 相关代码里甚至直接写了 `TODO: Make this work with non-assisted grasping`，说明这套 predicate 语义主要是围绕
assisted / sticky grasp 设计的。
- 但当前 Franka 配置使用的是 `physical` grasping mode。
- 这意味着现在的 success predicate 本身就可能和实际物理 grasp runtime 存在 contract mismatch；即使 goal 被判成
功，也不等价于 agent 按我们预期的方式完成了 retrieval。
- 另外，默认 BehaviorTask / wrapper 路径里 LTL 状态主要通过 `info["ltl"]` 暴露，不会天然自动变成 hard fail
termination；如果下游 eval 没显式把 safety 作为 hard success gate，shortcut success 还可能被进一步放大。

#### 第五，问题本质

- 这组任务当前的真实语义更接近：
`safe grasp bottom target under a stack`
- 而不是：
`clear stacked/container-associated objects, then safely retrieve the target receptacle`
- 因此这不是 “agent 会不会投机” 的问题，而是 benchmark / task spec 本身给了一个 spec-compliant shortcut。
- 这个问题应明确归类为 task design / benchmark semantics bug，而不是 policy failure。

## 04_stack_flat_20260407

### Common issues

- Camera / view contract problems
  - cam 镜头角度仍有问题，一些 scene 没有稳定把 Franka + tabletop objects 一起框进来

- Flat / thin object placement robustness problems
  - `stack_flat` 这组比前几组更容易暴露 flat / thin object 的采样和摆放问题
  - very flat object（如 `map`、`wax_paper`）容易卡在 `initial kinematic condition sampling`
  - 桌子 shape / selected region 不合理时，也更容易出现物体悬空、初始摆放不稳、或 Franka placement 不合理
  - `grocery_store_half_stocked`：桌子形状导致物体直接悬空，Franka 摆放位置也明显离谱

- 大面积 run 不完整 / only-log
  - 这组有大面积只剩 `stdout.log` 的 task，共 22 个，是目前这几组里 only-log 比例最高的一组
  - 不是单一 root cause，而是 mixed failure set，主要分 4 类：
    - `timeout / init stall`
      - 如 `Beechwood_0_garden`, `Ihlen_1_int`, `Rs_garden`, `Wainscott_0_garden`, `hall_conference_large`, `house_single_floor`, `office_cubicles_left`, `office_large`, `restaurant_brunch`
      - 部分 log 明显卡在 very flat object 的 `ontop` sampling，其余多数更像 scene init / task init 卡住后被 300s timeout kill
    - `ClothPrim + flatcache incompatibility`
      - `Merom_1_int`, `grocery_store_cafe`, `hall_glass_ceiling`, `restaurant_asian`, `restaurant_cafeteria`
      - 统一报 `AssertionError: Cannot use flatcache with ClothPrim!`，随后 segfault
    - `support / object mapping failure`
      - `house_double_floor_lower`: `No suitable table-like surface found in scene`
      - `office_bike`: sampled `magazine` 没有成功 map 到 simulator object，随后 `NoneType.exists` crash
    - `GPU / renderer OOM`
      - `restaurant_hotel`, `restaurant_urban`, `school_biology`, `school_chemistry`, `school_computer_lab_and_infirmary`, `school_geography`
      - 统一是 Vulkan / RTX `ERROR_OUT_OF_DEVICE_MEMORY`

- Partial artifact save / runtime interrupted
  - `grocery_store_asian` 不是纯 only-log，但也明显是 run 不完整：目录里只有 3 个 side-view mp4 + `stdout.log`，缺主 `rollout_ep1.mp4`、`diagnostics.jsonl`、`scene_ep1.json`
  - 结合 video review，更像是 runtime 中途出问题 / 保存不完整，而不是正常结束

- Overall pattern
  - 这组的共性更偏 flat-object task 本身带来的 runtime / asset compatibility / sampling robustness 问题
  - 相比前几组，除了 camera 之外，更核心的是 flat / thin object 对 scene init、support sampling、artifact 保存完整性都更敏感

## 05_transfer_1200s_20260408

### Common issues

- Camera / view contract problems
  - 相机位姿仍有问题，一些 scene 没有稳定把 Franka + source/dest/food 一起框进来

- Placement budget / clearance problems
  - 当前摆放看起来更像主要考虑了 footprint，没有充分考虑实际高度、体积、开口尺寸，以及和 Franka 的 clearance
  - `gates_bedroom`：bowl 直接被 Franka 卡住，simulation 开始会先发生轻微 collision，object 会弹跳一下
  - `hall_conference_large`：source 和 dest 都偏大，起始摆放彼此太近，后续执行阶段容易 collision
  - `Rs_garden`：dest 篮子体积太大、底部占地又偏小，code 可能按底面积误判为“可放”，结果实际是倾斜摆放，并且直接和 Franka 挤在一起

- Initial placement height / contact problems
  - 一些 object 的初始 z 没有真正落实到桌面接触上，simulation 开始后会短距离自由落体、砸到桌面，或者直接起飞
  - `grocery_store_cafe`, `Pomaria_0_garden`：object 从高处直接砸到桌面
  - `house_double_floor_lower`, `Ihlen_0_int`：物体摆放位置没真正落在桌子上，simulation 开始直接起飞
  - `Wainscott_0_int`：source / dest / food 初始摆放互相冲突，导致 simulation 开始后直接 collision，或者短距离自由落体到彼此 object 上面 / 桌面上

- Support geometry problems
  - 桌子 shape / AABB 中空问题仍然存在，会导致 source / dest / food 悬空、摆放不稳，或者后续接触状态异常

- Object compatibility / task semantics problems
  - 有些 source / dest candidate 在 `transfer` 语义上不太合理，开口、深度、稳定性和 food 尺寸不匹配
  - `Pomaria_1_int`：banana 对一个开口很小的 container，`inside` 很难合理成立；如果退化成 `ontop` 又非常不稳定，food 很容易掉下来。这类 object candidate 需要考虑是否从 transfer pool 里删掉

- Only-log scenes: log-confirmed causes
  - 这组真正只有 `stdout.log` 的 task 只有 6 个，不是 timeout，也不是 GPU OOM 主导，而是 task-generation / scene semantics / food taxonomy coverage 问题
  - `Beechwood_0_int`, `Ihlen_1_int`, `Merom_1_int`
    - 共同报错：`Missing valid object models for all categories: ['toast']`
    - 随后 `toast.n.01_1 and/or <dest> are not mapped to simulator objects in scope`
    - 最后在 `behavior_task.reset()` 里因为 `obj is None` 触发 `AttributeError: 'NoneType' object has no attribute 'exists'`，再 segfault
    - root cause 更像是 `toast` 这类 food category 没有有效 object model / 没成功 import 到 simulator scope
  - `hotel_gym_spa`
    - `RuntimeError: No suitable table-like surface found in scene`
    - 是 support discovery failure，不是 server crash
  - `office_cubicles_left`
    - `Room type [meeting_room] ... do not contain or cannot sample all the objects needed`
    - 随后 `RuntimeError: stack expects a non-empty TensorList`，最后 segfault
    - 本质是 presampling 已经失败，后续 task state 为空
  - `restaurant_brunch`
    - `AssertionError: Invalid synset: croissant.n.01`
    - 说明 food synset 本身不在当前 BDDL object taxonomy 里，初始化 sampler 时就挂掉了

- Overall pattern
  - `transfer` 这组的共性，不像前几组那样主要是 OOM / timeout，而是：
    - source / dest / food 的几何摆放和 clearance 设计不稳
    - container 开口 / 容量 / 形状与 food 尺寸不匹配
    - food-related taxonomy / object model / simulator mapping robustness 不够
