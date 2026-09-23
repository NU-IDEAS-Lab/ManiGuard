"""Family-specific motion sequences for scripted demonstration collection.

FamilySkeleton implementations provide grasp candidates, motion segments, live
target resolution, and optional acceptance checks. FAMILY maps driver names to
the clutter, cabinet, cabinet_firsthalf, stack, jar, dusty, and lid skeletons.
The shared executor performs trajectory execution, recording, and goal/LTL checks.
"""
from maniguard.data.datagen.families.cabinet import CabinetSkeleton
from maniguard.data.datagen.families.cabinet_firsthalf import CabinetFirstHalfSkeleton
from maniguard.data.datagen.families.clutter import ClutterSkeleton
from maniguard.data.datagen.families.dusty import DustySkeleton
from maniguard.data.datagen.families.jar import JarSkeleton
from maniguard.data.datagen.families.lid import LidSkeleton
from maniguard.data.datagen.families.stack import StackSkeleton

FAMILY = {
    "clutter": ClutterSkeleton,
    "cabinet": CabinetSkeleton,
    # Truncated-horizon variant of `cabinet` (phases 1-2 only); needs --horizon-override.
    "cabinet_firsthalf": CabinetFirstHalfSkeleton,
    "stack": StackSkeleton,
    "jar": JarSkeleton,
    "dusty": DustySkeleton,
    "lid": LidSkeleton,
}

