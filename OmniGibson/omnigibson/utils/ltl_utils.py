from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import spot
import buddy
import numpy as np

from bddl.logic_base import BinaryAtomicFormula, UnaryAtomicFormula
from omnigibson.utils.bddl_utils import SUPPORTED_PREDICATES


class PropositionType(Enum):
    UNARY_STATE = "unary_state"
    BINARY_RELATION = "binary_relation"


@dataclass
class AtomicProposition:
    name: str
    type: PropositionType
    args: Tuple
    eval_fn: Callable
    description: str = ""
    category: str = ""

    def evaluate(self, env=None) -> bool:
        try:
            return bool(self.eval_fn(env))
        except Exception:
            return False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, AtomicProposition) and self.name == other.name

    def __repr__(self):
        return f"Prop({self.name})"


class PropositionSet:
    def __init__(self, name: str = "default"):
        self.name = name
        self.propositions: List[AtomicProposition] = []
        self.prop_dict: Dict[str, AtomicProposition] = {}
        self.categories: Dict[str, List[str]] = {}

    def add_proposition(self, prop: AtomicProposition) -> None:
        if prop.name not in self.prop_dict:
            self.propositions.append(prop)
            self.prop_dict[prop.name] = prop
            if prop.category:
                self.categories.setdefault(prop.category, []).append(prop.name)

    def get_proposition(self, name: str) -> Optional[AtomicProposition]:
        return self.prop_dict.get(name)

    def get_propositions_by_category(self, category: str) -> List[AtomicProposition]:
        names = self.categories.get(category, [])
        return [self.prop_dict[name] for name in names]

    def get_label(self, env=None) -> np.ndarray:
        return np.array([prop.evaluate(env) for prop in self.propositions], dtype=bool)

    def get_label_dict(self, env=None) -> Dict[str, bool]:
        return {prop.name: prop.evaluate(env) for prop in self.propositions}

    def __len__(self) -> int:
        return len(self.propositions)

    def __iter__(self):
        return iter(self.propositions)

    def __getitem__(self, idx: int) -> AtomicProposition:
        return self.propositions[idx]

    def __repr__(self):
        return f"PropositionSet({self.name}, {len(self)} propositions)"


class AtomicPropositionGenerator:
    def __init__(self, task, verbose: bool = False):
        self.task = task
        self.backend = task.backend
        self.object_scope = task.object_scope
        self.object_map = task.activity_conditions.parsed_objects
        self.verbose = verbose
        self.propositions = PropositionSet(name=task.activity_name)

    def generate_all(self, include_goals: bool = False) -> PropositionSet:
        self._generate_unary_propositions()
        self._generate_binary_propositions()
        return self.propositions

    def _generate_unary_propositions(self) -> None:
        for pred_name, pred_cls in SUPPORTED_PREDICATES.items():
            if not issubclass(pred_cls, UnaryAtomicFormula):
                continue
            for obj_inst in self.object_scope.keys():
                prop_name = f"{obj_inst}_{pred_name}"
                prop = self._create_unary_proposition(prop_name, pred_name, pred_cls, obj_inst)
                self.propositions.add_proposition(prop)

    def _generate_binary_propositions(self) -> None:
        obj_insts = list(self.object_scope.keys())
        for pred_name, pred_cls in SUPPORTED_PREDICATES.items():
            if not issubclass(pred_cls, BinaryAtomicFormula):
                continue
            for obj_1 in obj_insts:
                for obj_2 in obj_insts:
                    if obj_1 == obj_2:
                        continue
                    prop_name = f"{obj_1}_{pred_name}_{obj_2}"
                    prop = self._create_binary_proposition(prop_name, pred_name, pred_cls, obj_1, obj_2)
                    self.propositions.add_proposition(prop)

    def _create_unary_proposition(self, prop_name: str, pred_name: str, pred_cls, obj_inst: str) -> AtomicProposition:
        pred = pred_cls(
            self.object_scope,
            self.backend,
            [obj_inst],
            self.object_map,
            generate_ground_options=False,
        )

        def eval_fn(_env=None, predicate=pred):
            try:
                return predicate.evaluate()
            except Exception:
                return False

        return AtomicProposition(
            name=prop_name,
            type=PropositionType.UNARY_STATE,
            args=(obj_inst, pred_name),
            eval_fn=eval_fn,
            description=f"{pred_name}({obj_inst})",
            category="unary_state",
        )

    def _create_binary_proposition(
        self, prop_name: str, pred_name: str, pred_cls, obj_1: str, obj_2: str
    ) -> AtomicProposition:
        pred = pred_cls(
            self.object_scope,
            self.backend,
            [obj_1, obj_2],
            self.object_map,
            generate_ground_options=False,
        )

        def eval_fn(_env=None, predicate=pred):
            try:
                return predicate.evaluate()
            except Exception:
                return False

        return AtomicProposition(
            name=prop_name,
            type=PropositionType.BINARY_RELATION,
            args=(obj_1, pred_name, obj_2),
            eval_fn=eval_fn,
            description=f"{pred_name}({obj_1}, {obj_2})",
            category="binary_relation",
        )

    def _condition_to_prop_name(self, cond) -> Tuple[Optional[str], bool]:
        negated = False
        if isinstance(cond, list) and cond and cond[0] == "not":
            negated = True
            cond = cond[1]
        if not isinstance(cond, list) or len(cond) < 2:
            return None, negated
        pred = cond[0]
        args = cond[1:]
        if len(args) == 1:
            return f"{args[0]}_{pred}", negated
        if len(args) == 2:
            return f"{args[0]}_{pred}_{args[1]}", negated
        return None, negated

    def _flatten_head_conditions(self, head) -> List[list]:
        if not head.flattened_condition_options:
            return []
        return head.flattened_condition_options[0]

    def get_grounded_goal_options(self) -> List[List[Tuple[str, bool]]]:
        options = []
        goal_options = self.task.ground_goal_state_options or []
        for option in goal_options:
            grounded = []
            for head in option:
                for cond in self._flatten_head_conditions(head):
                    prop_name, negated = self._condition_to_prop_name(cond)
                    if prop_name:
                        grounded.append((prop_name, negated))
            options.append(grounded)
        return options


class LTLLabelingFunction:
    def __init__(self, proposition_set: PropositionSet):
        self.prop_set = proposition_set
        self.num_propositions = len(proposition_set)
        self.prop_index = {prop.name: i for i, prop in enumerate(proposition_set)}

    def evaluate(self, env=None):
        label_array = self.prop_set.get_label(env)
        label_dict = self.prop_set.get_label_dict(env)
        return label_array, label_dict

    def evaluate_formula(self, env, formula: str) -> bool:
        label_dict = self.prop_set.get_label_dict(env)
        formula = formula.strip()
        for prop_name, value in label_dict.items():
            formula = formula.replace(prop_name, "True" if value else "False")
        formula = formula.replace("~", "not ")
        formula = formula.replace("&", "and")
        formula = formula.replace("|", "or")
        try:
            return bool(eval(formula))
        except Exception:
            return False

    def __repr__(self):
        return f"LTLLabeler({self.num_propositions} propositions)"


class LTLMonitor:
    def __init__(self, formula: str, translate_opts: Optional[Tuple[str, ...]] = ('monitor', 'det', 'complete')):
        self.formula_str = formula
        if translate_opts:
            opts = list(translate_opts)
            if "complete" not in opts:
                opts.append("complete")
            self.translate_opts = tuple(opts)
        else:
            self.translate_opts = translate_opts
        self._formula = None
        self._automaton = None
        self._dict = None
        self._ap_list: List[str] = []
        self._state = None
        self._can_reach_accepting = None

        self._initialize_spot()

    @property
    def ap_list(self) -> List[str]:
        return list(self._ap_list)

    @property
    def state(self):
        return self._state

    def reset(self) -> None:
        if self._automaton is not None:
            self._state = self._automaton.get_init_state_number()
            self._can_reach_accepting = None


    def _initialize_spot(self) -> None:
        
        self._formula = spot.formula(self.formula_str)
        self._ap_list = sorted(str(ap) for ap in spot.atomic_prop_collect(self._formula))

        try:
            if self.translate_opts:
                self._automaton = spot.translate(self._formula, *self.translate_opts)
            else:
                self._automaton = spot.translate(self._formula)
        except Exception:
            raise ValueError(f"Failed to translate LTL formula: {self.formula_str} into automaton.")

        self._dict = self._automaton.get_dict()
        # for ap in self._ap_list:
        #     self._register_ap(ap)
        self._state = self._automaton.get_init_state_number()
        
        
    def _check_ap_in_dict(self, ap: str) -> int:
        var = self._dict.varnum(ap)
        if var < 0:
            self._register_ap(ap)
            var = self._dict.varnum(ap)
        if var < 0:
            raise RuntimeError(f"Failed to register AP {ap}")
        return var
    
    
    def extract_ap_labels(self, label_dict: Dict[str, bool]) -> Dict[str, bool]:
        return {ap: bool(label_dict.get(ap, False)) for ap in self._ap_list}


    def _build_condition_bdd(self, label_dict: Dict[str, bool]):
        cond = buddy.bddtrue
        for ap in self._ap_list:
            var = self._check_ap_in_dict(ap)
            val = bool(label_dict.get(ap, False))
            cond = cond & (buddy.bdd_ithvar(var) if val else buddy.bdd_nithvar(var))
        return cond


    def _register_ap(self, ap: str) -> None:
        self._dict.register_proposition(spot.formula(ap), self)
        

    def step(self, label_dict: Dict[str, bool]) -> Dict[str, object]:
        if self._automaton is None:
            return {"state": None, "accepting": False}

        cond = self._build_condition_bdd(label_dict)
        next_state = None
        for transition in self._automaton.out(self._state):
            # Simplification of implies: A -> B  is equivalent to A & ~B == False
            if (cond & buddy.bdd_not(transition.cond)) == buddy.bddfalse:
                next_state = transition.dst
                break
        self._state = next_state
        accepting = self._automaton.state_is_accepting(self._state)
        return {
            "state": self._state,
            "accepting": accepting,
            "ap": self.extract_ap_labels(label_dict),
            "doomed": self.is_doomed_state(),
        }


    def is_doomed_state(self) -> bool:
        if self._automaton is None or self._state is None:
            return False
        if self._can_reach_accepting is None:
            self._can_reach_accepting = self._compute_accepting_reachability()
        return not self._can_reach_accepting.get(self._state, False)


    def _compute_accepting_reachability(self) -> Dict[int, bool]:
        num_states = int(self._automaton.num_states())
        graph = {s: [] for s in range(num_states)}
        reverse = {s: [] for s in range(num_states)}
        for s in range(num_states):
            for t in self._automaton.out(s):
                graph[s].append(t.dst)
                reverse[t.dst].append(s)

        accepting_states = set()
        for s in range(num_states):
            if self._automaton.state_is_accepting(s):
                accepting_states.add(s)

        # Tarjan SCC
        index = 0
        stack = []
        on_stack = set()
        indices = {}
        lowlink = {}
        sccs = []

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)

        for v in range(num_states):
            if v not in indices:
                strongconnect(v)

        accepting_scc_states = set()
        for comp in sccs:
            if any(s in accepting_states for s in comp):
                accepting_scc_states.update(comp)

        reachable = {s: False for s in range(num_states)}
        queue = list(accepting_scc_states)
        for s in queue:
            reachable[s] = True
        while queue:
            s = queue.pop()
            for p in reverse[s]:
                if not reachable[p]:
                    reachable[p] = True
                    queue.append(p)
        return reachable
