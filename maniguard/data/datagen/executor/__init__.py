"""Layer-2 generic executor (family-agnostic).

The reusable engine that plans / executes / gates / records / scales ANY family's
motion plan. Family-specific heuristics live in ``maniguard.data.datagen.families``;
the two sides meet only at ``contracts.FamilySkeleton`` + ``contracts.MotionSegment``,
so the engine never knows which family produced a plan and a family never touches
cuRobo / execution / the gate / the recorder.
"""
