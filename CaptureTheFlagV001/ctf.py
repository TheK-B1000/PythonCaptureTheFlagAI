from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import math
import time

# -------------------------------
# Utility geometry / data models
# -------------------------------
@dataclass
class Pose:
    x: float
    y: float

@dataclass
class Vehicle:
    name: str
    team: str        # e.g., "blue" or "red"
    vtype: str       # e.g., "mokai" or "heron"
    pose: Pose
    tagged_until: float = 0.0  # unix time when tag expires
    last_tag_time: float = -1e9  # last time this vehicle successfully tagged someone

def now() -> float:
    return time.time()

def dist(a: Pose, b: Pose) -> float:
    dx, dy = a.x - b.x, a.y - b.y
    return math.hypot(dx, dy)

def point_in_convex_polygon(p: Pose, poly: List[Tuple[float,float]]) -> bool:
    # Standard winding / half-space test for convex polys
    # Poly is list of (x,y) in CCW order.
    n = len(poly)
    if n < 3:
        return False
    prev = None
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        cross = (x2-x1)*(p.y-y1) - (y2-y1)*(p.x-x1)
        if prev is None:
            prev = cross
        else:
            if cross*prev < 0:
                return False
    return True

# -------------------------------
# Tag Manager (uFldTagManager)
# -------------------------------
@dataclass
class TagManagerConfig:
    tag_range: float = 25.0          # meters (default ~25 in docs)
    tag_duration: float = 30.0       # seconds (default 30s)
    tag_min_interval: float = 10.0   # seconds (default 10s)
    zone_one: List[Tuple[float,float]] = field(default_factory=list) # blue zone polygon
    zone_two: List[Tuple[float,float]] = field(default_factory=list) # red zone polygon
    team_one: str = "blue"
    team_two: str = "red"

class TagManager:
    def __init__(self, cfg: TagManagerConfig):
        self.cfg = cfg
        # vehicle_name -> Vehicle
        self.vehicles: Dict[str, Vehicle] = {}

    def update_node_report(self, name: str, team: str, vtype: str, pose: Pose):
        self.vehicles[name] = Vehicle(name, team, vtype, pose, 
                                      tagged_until=self.vehicles.get(name, Vehicle(name,team,vtype,pose)).tagged_until,
                                      last_tag_time=self.vehicles.get(name, Vehicle(name,team,vtype,pose)).last_tag_time)

    def _zone_for_team(self, team: str) -> List[Tuple[float,float]]:
        return self.cfg.zone_one if team == self.cfg.team_one else self.cfg.zone_two

    def is_tagged(self, v: Vehicle) -> bool:
        return now() < v.tagged_until

    def _in_own_zone(self, v: Vehicle) -> bool:
        return point_in_convex_polygon(v.pose, self._zone_for_team(v.team))

    def _cooldown_ready(self, v: Vehicle) -> bool:
        return now() - v.last_tag_time >= self.cfg.tag_min_interval

    def _eligible_targets(self, src: Vehicle) -> List[Vehicle]:
        opp_team = self.cfg.team_two if src.team == self.cfg.team_one else self.cfg.team_one
        candidates = []
        for v in self.vehicles.values():
            if v.team != opp_team: 
                continue
            if self.is_tagged(v): 
                continue
            # target must be outside its OWN zone
            if point_in_convex_polygon(v.pose, self._zone_for_team(v.team)):
                continue
            # distance check
            if dist(src.pose, v.pose) <= self.cfg.tag_range:
                candidates.append(v)
        return candidates

    def request_tag(self, src_name: str) -> Dict:
        # Return a result payload akin to TAG_RESULT_*
        if src_name not in self.vehicles:
            return {"event": "reject", "reason": "unknown_src"}

        src = self.vehicles[src_name]

        # 1) zone check for source
        if not self._in_own_zone(src):
            return {"event": "reject", "src": src.name, "team": src.team, "reason": "zone"}

        # 2) cooldown check
        if not self._cooldown_ready(src):
            return {"event": "reject", "src": src.name, "team": src.team, "reason": "freq"}

        # 3) candidate targets
        targets = self._eligible_targets(src)
        if not targets:
            return {"event": "ok", "src": src.name, "team": src.team, "tagged": None}

        # 4) nearest target
        target = min(targets, key=lambda t: dist(src.pose, t.pose))

        # 5) apply tag
        target.tagged_until = now() + self.cfg.tag_duration
        src.last_tag_time = now()
        return {"event": "ok", "src": src.name, "team": src.team, "tagged": target.name, "expires_at": target.tagged_until}

    def tick(self):
        # expire tags is implicit via time checks; nothing to do but could emit events
        pass

# -------------------------------
# Flag Manager (uFldFlagManager)
# -------------------------------
@dataclass
class Flag:
    label: str
    pose: Pose
    grab_range: float = 10.0
    owner: Optional[str] = None  # vehicle name if grabbed

class FlagManager:
    def __init__(self):
        self.flags: Dict[str, Flag] = {}
        self.vehicles: Dict[str, Vehicle] = {}

    def add_flag(self, label: str, x: float, y: float, grab_range: float = 10.0):
        self.flags[label] = Flag(label, Pose(x, y), grab_range)

    def update_node_report(self, name: str, team: str, vtype: str, pose: Pose):
        self.vehicles[name] = Vehicle(name, team, vtype, pose,
                                      tagged_until=self.vehicles.get(name, Vehicle(name,team,vtype,pose)).tagged_until,
                                      last_tag_time=self.vehicles.get(name, Vehicle(name,team,vtype,pose)).last_tag_time)

    def request_grab(self, vname: str) -> Dict:
        # Vehicle asks to grab any flags within range.
        if vname not in self.vehicles:
            return {"result": "deny", "reason": "unknown_src"}
        v = self.vehicles[vname]

        # In Aquaticus, a tagged vehicle cannot grab; this check is part of game mechanics.
        if now() < v.tagged_until:
            return {"result": "deny", "reason": "tagged"}

        grabbed = []
        for flag in self.flags.values():
            if flag.owner is not None:
                continue
            if dist(v.pose, flag.pose) <= flag.grab_range:
                flag.owner = vname
                grabbed.append(flag.label)

        if not grabbed:
            return {"result": "deny", "reason": "nothing_in_range"}
        return {"result": "ok", "grabbed": grabbed}

    def reset_flag_by_label(self, label: str):
        if label in self.flags:
            self.flags[label].owner = None

    def reset_flags_by_owner(self, vname: str):
        for f in self.flags.values():
            if f.owner == vname:
                f.owner = None

    def reset_all(self):
        for f in self.flags.values():
            f.owner = None

    def scored_goal(self, vname: str, home_zone_poly: List[Tuple[float,float]]) -> Optional[List[str]]:
        """
        Call this when the vehicle enters its home zone.
        If it carries flags and is untagged, it's a score: flags reset to 'ungrabbed'.
        """
        if vname not in self.vehicles:
            return None
        v = self.vehicles[vname]
        if now() < v.tagged_until:
            # Tagged before reaching home—lose flags
            self.reset_flags_by_owner(vname)
            return None
        if not point_in_convex_polygon(v.pose, home_zone_poly):
            return None

        delivered = []
        for f in self.flags.values():
            if f.owner == vname:
                delivered.append(f.label)
                f.owner = None
        return delivered


