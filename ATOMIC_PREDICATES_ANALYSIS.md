# Atomic Predicates in BDDL3: Definition & Evaluation with Physics Engine

## Overview
Atomic predicates in the BDDL3 framework are the fundamental logical units that evaluate object states and relationships. They are organized in a two-level hierarchy: **UnaryAtomicFormula** (single object) and **BinaryAtomicFormula** (two objects).

---

## 1. ATOMIC PREDICATE DEFINITIONS

### Location: [bddl3/bddl/logic_base.py](bddl3/bddl/logic_base.py)

#### Base Abstract Classes

**[AtomicFormula](bddl3/bddl/logic_base.py#L22-L25)** - Abstract base class
- Inherits from `Expression`
- Minimal initialization, provides structure for unary and binary predicates

**[UnaryAtomicFormula](bddl3/bddl/logic_base.py#L96-L148)** - Single-argument predicates
```python
class UnaryAtomicFormula(AtomicFormula):
    STATE_NAME = None
    
    @abstractmethod
    def _evaluate(self, obj):
        """Evaluate predicate for a single object"""
        pass
    
    @abstractmethod
    def _sample(self, obj, binary_state):
        """Sample/set predicate state"""
        pass
```

**Key methods:**
- `evaluate()` - Calls `_evaluate()` with object from scope
- `_evaluate(obj)` - Abstract method, must be implemented by subclasses
- `_sample(obj, binary_state)` - Abstract method for state modification

**Examples of unary predicates:**
- `cooked`, `frozen`, `open`, `folded`, `toggled_on`, `hot`, `on_fire`, `empty`, `broken`, `closed`, `real`, `future`

---

**[BinaryAtomicFormula](bddl3/bddl/logic_base.py#L28-L95)** - Two-argument predicates
```python
class BinaryAtomicFormula(AtomicFormula):
    STATE_NAME = None
    
    @abstractmethod
    def _evaluate(self, obj1, obj2):
        """Evaluate relationship between two objects"""
        pass
    
    @abstractmethod
    def _sample(self, obj1, obj2, binary_state):
        """Sample/set relationship state"""
        pass
```

**Key methods:**
- `evaluate()` - Calls `_evaluate()` with two objects from scope
- `_evaluate(obj1, obj2)` - Abstract method, must be implemented
- `_sample(obj1, obj2, binary_state)` - Abstract method for state modification

**Examples of binary predicates:**
- `ontop`, `inside`, `covered`, `filled`, `contains`, `touching`, `nextto`, `under`, `overlaid`, `attached`, `draped`, `grasped`, `saturated`

---

## 2. ATOMIC PREDICATE IMPLEMENTATIONS

### Location 1: [bddl3/bddl/trivial_backend.py](bddl3/bddl/trivial_backend.py)

**Trivial Backend** - Reference implementation with mock physics
- Maps predicate names to concrete classes via `PREDICATE_MAPPING`
- Implements all predicates as concrete subclasses of `UnaryAtomicFormula` or `BinaryAtomicFormula`

Example unary implementation:
```python
class TrivialCookedPredicate(UnaryAtomicFormula):
    STATE_NAME = "cooked"
    
    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_cooked())
        return obj.get_cooked()
    
    def _sample(self, obj1, binary_state):
        pass
```

Example binary implementation:
```python
class TrivialOnTopPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"
    
    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_ontop(obj2))
        return obj1.get_ontop(obj2)
    
    def _sample(self, obj1, obj2, binary_state):
        pass
```

---

### Location 2: [behavior-1k/OmniGibson/omnigibson/utils/bddl_utils.py](behavior-1k/OmniGibson/omnigibson/utils/bddl_utils.py)

**OmniGibson Backend** - Production implementation with actual physics engine integration

#### Generic Predicate Base Classes (L132-L171)

```python
class ObjectStateUnaryPredicate(UnaryAtomicFormula):
    STATE_CLASS = None  # Object state class to delegate to
    STATE_NAME = None
    
    def _evaluate(self, entity, **kwargs):
        # Delegates to entity's get_state method
        return entity.get_state(self.STATE_CLASS, **kwargs)
    
    def _sample(self, entity, binary_state, **kwargs):
        # Delegates to entity's set_state method
        return entity.set_state(self.STATE_CLASS, binary_state, **kwargs)

class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None  # Object state class to delegate to
    STATE_NAME = None
    
    def _evaluate(self, entity1, entity2, **kwargs):
        # Delegates to entity1's get_state method with entity2 as parameter
        return (
            entity1.get_state(self.STATE_CLASS, entity2.wrapped_obj, **kwargs)
            if (entity2.exists and entity2.initialized)
            else False
        )
    
    def _sample(self, entity1, entity2, binary_state, **kwargs):
        return (
            entity1.set_state(self.STATE_CLASS, entity2.wrapped_obj, binary_state, **kwargs)
            if (entity2.exists and entity2.initialized)
            else None
        )
```

#### Factory Functions (L176-L185)
```python
def get_unary_predicate_for_state(state_class, state_name):
    """Dynamically creates unary predicate class for a given object state"""
    return type(
        state_class.__name__ + "StateUnaryPredicate",
        (ObjectStateUnaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )

def get_binary_predicate_for_state(state_class, state_name):
    """Dynamically creates binary predicate class for a given object state"""
    return type(
        state_class.__name__ + "StateBinaryPredicate",
        (ObjectStateBinaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )
```

---

## 3. ATOMIC PREDICATE EVALUATION WITH PHYSICS ENGINE

### How the Evaluation Chain Works

```
Atomic Predicate (e.g., OnTopPredicate)
    ↓ calls evaluate()
Checks if objects are in scope
    ↓
Calls _evaluate(obj1, obj2)
    ↓
Delegates to entity.get_state(STATE_CLASS, obj2)
    ↓
Object State Implementation (e.g., OnTop class)
    ↓
Physics queries via PyBullet/PhysX
    ↓
Returns boolean result
```

### Object State Classes: [OmniGibson/omnigibson/object_states/](OmniGibson/omnigibson/object_states/)

The actual physics evaluation happens in object state classes:

#### Example 1: **Touching** - Contact-based Predicate
**File:** [OmniGibson/omnigibson/object_states/touching.py](OmniGibson/omnigibson/object_states/touching.py)

```python
class Touching(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @staticmethod
    def _check_contact(obj_a, obj_b):
        return len(set(obj_a.links.values()) & obj_b.states[ContactBodies].get_value()) > 0
    
    def _get_value(self, other):
        # Checks if any link of obj_a is in contact with obj_b
        # Uses ContactBodies state which queries physics engine contact list
        return self._check_contact(other, self.obj) and self._check_contact(self.obj, other)
```

**Physics Integration:**
- Uses `ContactBodies` state (L8)
- Queries contact information from physics engine
- Returns boolean indicating if objects are touching

#### Example 2: **Inside** - Volume-based Predicate
**File:** [OmniGibson/omnigibson/object_states/inside.py](OmniGibson/omnigibson/object_states/inside.py)

```python
class Inside(RelativeObjectState, KinematicsMixin, BooleanStateMixin):
    def _get_value(self, other):
        # Step 1: Get AABB (Axis-Aligned Bounding Box)
        aabb_lower, aabb_upper = self.obj.states[AABB].get_value()
        inner_object_pos = (aabb_lower + aabb_upper) / 2.0
        outer_object_aabb_lo, outer_object_aabb_hi = other.states[AABB].get_value()
        
        # Step 2: Quick AABB check
        if not (th.le(outer_object_aabb_lo, inner_object_pos).all() and ...):
            return False
        
        # Step 3: Detailed volume check using meta links
        for link in other.links.values():
            if link.meta_link_type in CONTAINER_META_LINK_TYPES:
                in_volume |= link.check_points_in_volume(points)
        
        return th.any(in_volume).item()
```

**Physics Integration:**
- Queries AABB from physics engine
- Uses container volume information from object links
- Performs point-in-volume tests

#### Example 3: **ContactBodies** - Low-level Contact Query
**File:** [OmniGibson/omnigibson/object_states/contact_bodies.py](OmniGibson/omnigibson/object_states/contact_bodies.py)

```python
class ContactBodies(AbsoluteObjectState):
    def _get_value(self, ignore_objs=None, non_zero_impulse=False):
        # Query physics engine for contact list
        bodies = set()
        for contact in self.obj.contact_list():
            if not non_zero_impulse or np.linalg.norm(tuple(contact.impulse)) > 0:
                bodies.update({contact.body0, contact.body1})
        
        # Filter out self bodies
        bodies -= set(self.obj.link_prim_paths)
        rigid_prims = prim_paths_to_rigid_prims(bodies, self.obj.scene)
        return {p for o, p in rigid_prims if ignore_objs is None or o not in ignore_objs}
```

**Physics Integration:**
- Calls `self.obj.contact_list()` - direct physics engine query
- Filters by impulse magnitude if needed
- Returns set of bodies in contact

---

## 4. PREDICATE PARSING & RESOLUTION

### Location: [bddl3/bddl/parsing.py](bddl3/bddl/parsing.py#L88-L108)

Predicates are parsed from domain definition files:

```python
def parse_predicates(group):
    """Parses predicates from domain PDDL file"""
    predicates = {}
    for pred in group:
        predicate_name = pred.pop(0)  # e.g., "ontop"
        arguments = {}  # e.g., {"?obj1": "object", "?obj2": "object"}
        # ... parse arguments and types
        predicates[predicate_name] = arguments
    return predicates
```

### Domain Files
- [bddl3/bddl/activity_definitions/domain_omnigibson.bddl](bddl3/bddl/activity_definitions/domain_omnigibson.bddl)
- [bddl3/bddl/activity_definitions/domain_igibson.bddl](bddl3/bddl/activity_definitions/domain_igibson.bddl)

These files define the predicate signatures.

---

## 5. PREDICATE EVALUATION FLOW

### Location: [bddl3/bddl/condition_evaluation.py](bddl3/bddl/condition_evaluation.py)

```python
class HEAD(Expression):
    def __init__(self, scope, backend, body, object_map, generate_ground_options=True):
        # Get the predicate class from backend
        predicate_class = get_predicate_for_token(subexpression[0], backend)
        
        # Instantiate the predicate
        self.children.append(
            predicate_class(
                scope,
                backend,
                subexpression[1:],  # arguments
                object_map,
                generate_ground_options=generate_ground_options,
            )
        )
    
    def evaluate(self):
        # Call evaluate on the predicate
        self.child_values = [child.evaluate() for child in self.children]
        return self.child_values[0]

def get_predicate_for_token(token, backend):
    """Resolves predicate name to class"""
    if token in TOKEN_MAPPING:  # Special predicates like AND, OR, etc.
        return TOKEN_MAPPING[token]
    else:
        try:
            return backend.get_predicate_class(token)
        except KeyError as e:
            raise UnsupportedPredicateError(e)
```

---

## 6. BACKEND ABSTRACTION

### Location: [bddl3/bddl/backend_abc.py](bddl3/bddl/backend_abc.py)

```python
class BDDLBackend(with_metaclass(ABCMeta)):
    @abstractmethod
    def get_predicate_class(self, predicate_name):
        """Given predicate_name, return an implementation of 
        bddl.logic_base.AtomicFormula or subclasses."""
        pass
```

Different backends (Trivial, OmniGibson) implement this interface to provide their own predicate implementations.

---

## Summary Architecture

```
Domain BDDL File
    ↓ (parsed by parsing.py)
Predicate Definition (name → argument types)
    ↓
Condition Evaluation (condition_evaluation.py)
    ↓ (resolves via backend)
Backend.get_predicate_class(name)
    ↓
Atomic Formula Class (logic_base.py)
    ├─ UnaryAtomicFormula
    └─ BinaryAtomicFormula
    ↓ (instance created with scope, body)
    ↓ evaluate()
    ↓ _evaluate(obj1, obj2) or _evaluate(obj)
    ↓ (OmniGibson backend)
    ├─ ObjectStateUnaryPredicate
    │   └─ entity.get_state(STATE_CLASS)
    │       └─ Object State Class (object_states/*.py)
    │           └─ Physics Engine Query (contact, AABB, volume, etc.)
    │               └─ Boolean Result
    └─ ObjectStateBinaryPredicate
        └─ entity1.get_state(STATE_CLASS, entity2)
            └─ Object State Class (object_states/*.py)
                └─ Physics Engine Query
                    └─ Boolean Result
```

---

## Key Physics Engine Integration Points

1. **Contact Detection** - [object_states/contact_bodies.py](OmniGibson/omnigibson/object_states/contact_bodies.py)
   - `contact_list()` queries from PhysX physics engine

2. **AABB Queries** - [object_states/aabb.py](OmniGibson/omnigibson/object_states/aabb.py)
   - Bounding box information for spatial checks

3. **Link/Body Geometry** - [object_states/object_state_base.py](OmniGibson/omnigibson/object_states/object_state_base.py)
   - Meta links for container volumes
   - Link collision checks

4. **Kinematics Mixin** - [object_states/kinematics_mixin.py](OmniGibson/omnigibson/object_states/kinematics_mixin.py)
   - Position/rotation queries from physics engine

5. **Simulator Contact Callback** - [OmniGibson/omnigibson/simulator.py](OmniGibson/omnigibson/simulator.py#L1261-L1264)
   - `_on_contact()` callback subscribes to physics contact events
   - Updates object states when contact occurs
