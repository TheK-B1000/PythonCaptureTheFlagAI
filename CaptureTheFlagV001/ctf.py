from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, DefaultDict
from collections import defaultdict, deque
import math, random, time, threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ------------------------------- Helpers
@dataclass
class Pose:
    x: float
    y: float

@dataclass
class Vehicle:
    name: str
    team: str        # "blue" or "red"
    vtype: str       # "mokai" or "heron"
    pose: Pose
    tagged_until: float = 0.0
    last_tag_time: float = -1e9
    carrying: List[str] = field(default_factory=list)  # labels of flags

def now() -> float:
    return time.time()

def dist(a: Pose, b: Pose) -> float:
    dx, dy = a.x - b.x, a.y - b.y
    return math.hypot(dx, dy)

def point_in_convex_polygon(p: Pose, poly: List[Tuple[float,float]]) -> bool:
    n = len(poly)
    if n < 3: return False
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

# ------------------------------- Tag Manager
@dataclass
class TagManagerConfig:
    tag_range: float = 25.0
    tag_duration: float = 30.0
    tag_min_interval: float = 10.0
    zone_one: List[Tuple[float,float]] = field(default_factory=list) # blue half
    zone_two: List[Tuple[float,float]] = field(default_factory=list) # red half
    team_one: str = "blue"
    team_two: str = "red"

class TagManager:
    def __init__(self, cfg: TagManagerConfig):
        self.cfg = cfg
        self.vehicles: Dict[str, Vehicle] = {}
        self._event_id = 0

    def update_node_report(self, name: str, team: str, vtype: str, pose: Pose):
        # ensure we have a Vehicle instance in `v`
        if name in self.vehicles:
            v = self.vehicles[name]
            v.pose = pose
            v.team = team
            v.vtype = vtype
        else:
            v = Vehicle(name, team, vtype, pose)
            self.vehicles[name] = v

        # Tag if outside the overall field (not in either team's half polygon)
        in_zone1 = point_in_convex_polygon(pose, self.cfg.zone_one)
        in_zone2 = point_in_convex_polygon(pose, self.cfg.zone_two)
        if not (in_zone1 or in_zone2):
            v.tagged_until = now() + self.cfg.tag_duration

    def _zone_for_team(self, team: str) -> List[Tuple[float,float]]:
        return self.cfg.zone_one if team == self.cfg.team_one else self.cfg.zone_two

    def is_tagged(self, v: Vehicle) -> bool:
        return now() < v.tagged_until

    def _in_own_half(self, v: Vehicle) -> bool:
        # Use field midline (max x of blue zone) to split halves
        mid_x = max(x for x, _ in self.cfg.zone_one)
        return v.pose.x <= mid_x if v.team == self.cfg.team_one else v.pose.x >= mid_x

    def _cooldown_ready(self, v: Vehicle) -> bool:
        return now() - v.last_tag_time >= self.cfg.tag_min_interval

    def _eligible_targets(self, src: Vehicle) -> List[Vehicle]:
        opp_team = self.cfg.team_two if src.team == self.cfg.team_one else self.cfg.team_one
        out = []
        for v in self.vehicles.values():
            if v.team != opp_team:
                continue
            if self.is_tagged(v):
                continue
            # Tagging is only allowed if src is in own half *and* target is in src's half (i.e., defending)
            if not self._in_own_half(src) or self._in_own_half(v):
                continue
            if dist(src.pose, v.pose) <= self.cfg.tag_range:
                out.append(v)
        return out

    def request_tag(self, src_name: str) -> Dict:
        self._event_id += 1
        if src_name not in self.vehicles:
            return {"event": self._event_id, "result": "reject", "reason": "unknown_src"}

        src = self.vehicles[src_name]
        if self.is_tagged(src):
            return {"event": self._event_id, "src": src.name, "team": src.team, "result": "reject", "reason": "tagged"}
        if not self._in_own_half(src):
            return {"event": self._event_id, "src": src.name, "team": src.team, "result": "reject", "reason": "zone"}
        if not self._cooldown_ready(src):
            return {"event": self._event_id, "src": src.name, "team": src.team, "result": "reject", "reason": "freq"}

        cands = self._eligible_targets(src)
        if not cands:
            return {"event": self._event_id, "src": src.name, "team": src.team, "result": "ok", "tagged": None}

        tgt = min(cands, key=lambda t: dist(src.pose, t.pose))
        tgt.tagged_until = now() + self.cfg.tag_duration
        src.last_tag_time = now()
        return {"event": self._event_id, "src": src.name, "team": src.team, "result": "ok", "tagged": tgt.name, "expires_at": tgt.tagged_until}

    def tick(self):
        pass

# ------------------------------- Flag Manager
@dataclass
class Flag:
    label: str
    pose: Pose
    grab_range: float = 10.0
    owner: Optional[str] = None  # vehicle name

class FlagManager:
    def __init__(self):
        self.flags: Dict[str, Flag] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.bus = None  # set externally to World's bus

    def add_flag(self, label: str, x: float, y: float, grab_range: float = 10.0):
        self.flags[label] = Flag(label, Pose(x, y), grab_range)

    def update_node_report(self, name: str, team: str, vtype: str, pose: Pose):
        if name in self.vehicles:
            v = self.vehicles[name]
            v.pose = pose
            v.team = team
            v.vtype = vtype
        else:
            self.vehicles[name] = Vehicle(name, team, vtype, pose)

    def request_grab(self, vname: str) -> Dict:
        if vname not in self.vehicles:
            return {"result": "deny", "reason": "unknown_src"}
        v = self.vehicles[vname]
        if now() < v.tagged_until:
            return {"result": "deny", "reason": "tagged"}

        opp_flag_label = "red_flag" if v.team == "blue" else "blue_flag"
        flag = self.flags[opp_flag_label]

        if dist(v.pose, flag.pose) <= flag.grab_range and flag.owner is None:
            flag.owner = vname
            v.carrying.append(flag.label)
            return {"result": "ok", "grabbed": [flag.label]}
        return {"result": "deny", "reason": "nothing_in_range"}

    def reset_flag_by_label(self, label: str):
        if label in self.flags:
            owner = self.flags[label].owner
            if owner and owner in self.vehicles:
                try:
                    self.vehicles[owner].carrying.remove(label)
                except ValueError:
                    pass
            self.flags[label].owner = None

    def reset_flags_by_owner(self, vname: str):
        for f in self.flags.values():
            if f.owner == vname:
                f.owner = None
        if vname in self.vehicles:
            self.vehicles[vname].carrying.clear()

    # NEW: scoring by home-circle (5 m radius default from World)
    def scored_goal(self, vname: str, home_center: Pose, home_radius: float) -> Optional[List[str]]:
        if vname not in self.vehicles:
            return None
        v = self.vehicles[vname]
        if now() < v.tagged_until:
            self.reset_flags_by_owner(vname)
            return None
        if dist(v.pose, home_center) > home_radius:
            return None
        delivered = list(v.carrying)
        if delivered:
            self.reset_flags_by_owner(vname)
            if self.bus:
                self.bus.publish("score", {"team": v.team, "flags": delivered})
        return delivered

# ------------------------------- Tiny async bus (pub/sub)
class AsyncBus:
    def __init__(self):
        self._subs: DefaultDict[str, List[Callable[[dict], None]]] = defaultdict(list)
        self._queue: "deque[Tuple[str,dict]]" = deque()
        self._lock = threading.Lock()

    def subscribe(self, topic: str, cb: Callable[[dict], None]):
        self._subs[topic].append(cb)

    def publish(self, topic: str, payload: dict):
        with self._lock:
            self._queue.append((topic, payload))

    def drain(self, max_msgs: int = 1000):
        processed = 0
        while processed < max_msgs:
            with self._lock:
                if not self._queue:
                    break
                topic, payload = self._queue.popleft()
            for cb in self._subs.get(topic, []):
                cb(payload)
            processed += 1

# ------------------------------- World
@dataclass
class AgentCtrl:
    max_speed: float = 2.0  # m/s for heron, mokai adjusted later
    wander: float = 0.1     # random heading jitter
    goal_gain: float = 2.0  # proportional steering

class World:
    def __init__(self, width=160.0, height=80.0, dt=0.5):
        self.w = width
        self.h = height
        self.dt = dt
        self.bus = AsyncBus()

        self.blue_poly = [(0,0),(self.w/2,0),(self.w/2,self.h),(0,self.h)]
        self.red_poly  = [(self.w/2,0),(self.w,0),(self.w,self.h),(self.w/2,self.h)]

        self.tag_mgr = TagManager(TagManagerConfig(
            zone_one=self.blue_poly, zone_two=self.red_poly,
            team_one="blue", team_two="red"
        ))
        self.flag_mgr = FlagManager()
        self.flag_mgr.bus = self.bus

        self.flags = {
            "blue_flag": Pose(x=20, y=self.h/2),
            "red_flag": Pose(x=self.w-20, y=self.h/2),
        }
        self.flag_mgr.add_flag("blue_flag", self.flags["blue_flag"].x, self.flags["blue_flag"].y, grab_range=10.0)
        self.flag_mgr.add_flag("red_flag", self.flags["red_flag"].x, self.flags["red_flag"].y, grab_range=10.0)

        self.vehicles: Dict[str, Vehicle] = {}
        self._spawn_team("blue", x=self.w*0.25, ymid=self.h*0.5)
        self._spawn_team("red",  x=self.w*0.75, ymid=self.h*0.5)

        self.ctrl: Dict[str, AgentCtrl] = {}
        self.heading: Dict[str, float] = {}
        self.role: Dict[str, str] = {}  # 'attacker' or 'defender'

        for name, v in self.vehicles.items():
            self.ctrl[name] = AgentCtrl(max_speed=1.8 if v.vtype=="heron" else 4.0)
            self.heading[name] = random.uniform(-math.pi, math.pi)
            # Role assignment: herons defend, mokai attack
            self.role[name] = 'defender' if v.vtype == 'heron' else 'attacker'

        self.home_radius = 5.0  # 10 m diameter scoring circle

        # Optional: subscribe to scores (print to console)
        self.bus.subscribe("score", lambda p: print(f"[SCORE] Team {p['team']} delivered {p['flags']}"))

    def _spawn_team(self, team: str, x: float, ymid: float):
        y_offsets = [-20, -10, 10, 20]  # Spread 4 vehicles across home zone
        for i, dy in enumerate(y_offsets):
            vtype = "heron" if i < 2 else "mokai"  # First 2 are heron, last 2 are mokai
            name = f"{team[0]}{i+1}"
            v = Vehicle(name=name, team=team, vtype=vtype, pose=Pose(x, ymid+dy))
            self.vehicles[name] = v
            self.tag_mgr.update_node_report(name, team, vtype, v.pose)
            self.flag_mgr.update_node_report(name, team, vtype, v.pose)

    def in_bounds(self, p: Pose) -> bool:
        return 0 <= p.x <= self.w and 0 <= p.y <= self.h

    def _goal_for(self, v: Vehicle) -> Pose:
        if v.team == "blue":
            enemy_flag = self.flags["red_flag"]
            home_flag  = self.flags["blue_flag"]
        else:
            enemy_flag = self.flags["blue_flag"]
            home_flag  = self.flags["red_flag"]

        # If carrying, always go home
        if v.carrying:
            return home_flag

        role = self.role.get(v.name, 'attacker')
        if role == 'attacker':
            # Push toward enemy flag when not carrying
            return enemy_flag

        # Defender logic
        mid = self.w / 2.0
        def in_my_half(p: Pose, team: str) -> bool:
            return p.x <= mid if team == "blue" else p.x >= mid

        # Enemies inside our half → defend closest to our flag
        enemies = [e for e in self.vehicles.values() if e.team != v.team and in_my_half(e.pose, v.team)]
        if enemies:
            target = min(enemies, key=lambda e: dist(e.pose, home_flag))
            return target.pose

        # No invaders: patrol near home flag (slight jitter to avoid clumping)
        jitter = 4.0
        return Pose(home_flag.x + random.uniform(-jitter, jitter),
                    home_flag.y + random.uniform(-jitter, jitter))

    def _own_zone_poly(self, team: str):
        return self.blue_poly if team == "blue" else self.red_poly

    def _update_motion(self, name: str):
        v = self.vehicles[name]
        ctrl = self.ctrl[name]
        tgt = self._goal_for(v)
        desired = math.atan2(tgt.y - v.pose.y, tgt.x - v.pose.x)
        self.heading[name] = self._ang_wrap(self.heading[name] + ctrl.goal_gain * (desired - self.heading[name]) * self.dt + random.uniform(-ctrl.wander, ctrl.wander) * self.dt)
        speed = ctrl.max_speed * (0.3 + 0.7 * random.random())

        # Limp-home behavior when tagged
        if now() < v.tagged_until:
            speed = 0.5  # slightly higher so they make progress
            home = self.flags["blue_flag" if v.team=="blue" else "red_flag"]
            desired = math.atan2(home.y - v.pose.y, home.x - v.pose.x)
            self.heading[name] = self._ang_wrap(self.heading[name] + 0.9 * (desired - self.heading[name]) * self.dt)

        dx = math.cos(self.heading[name]) * speed * self.dt
        dy = math.sin(self.heading[name]) * speed * self.dt
        v.pose.x += dx
        v.pose.y += dy

        # Hard clip at borders, and tag if they step out
        if v.pose.x < 0 or v.pose.x > self.w or v.pose.y < 0 or v.pose.y > self.h:
            v.tagged_until = now() + self.tag_mgr.cfg.tag_duration
            if v.pose.x < 0: v.pose.x = 0
            elif v.pose.x > self.w: v.pose.x = self.w
            if v.pose.y < 0: v.pose.y = 0
            elif v.pose.y > self.h: v.pose.y = self.h
            self.heading[name] = random.uniform(-math.pi, math.pi)

        # Optional leash to keep defenders in their half (uncomment to enforce)
        # if self.role.get(name) == 'defender':
        #     mid = self.w / 2.0
        #     if v.team == "blue" and v.pose.x > mid:
        #         v.pose.x = mid
        #     if v.team == "red" and v.pose.x < mid:
        #         v.pose.x = mid

    @staticmethod
    def _ang_wrap(a):
        while a <= -math.pi: a += 2*math.pi
        while a > math.pi: a -= 2*math.pi
        return a

    def _update_managers(self):
        # Sync managers with new poses
        for v in self.vehicles.values():
            self.tag_mgr.update_node_report(v.name, v.team, v.vtype, v.pose)
            self.flag_mgr.update_node_report(v.name, v.team, v.vtype, v.pose)

        # Tagging bias: defenders always tag in own half; attackers only if in own half & not carrying
        mid_x = self.w / 2.0
        def in_own_half(vh: Vehicle) -> bool:
            return vh.pose.x <= mid_x if vh.team == "blue" else vh.pose.x >= mid_x

        for v in self.vehicles.values():
            if self.role.get(v.name, 'attacker') == 'defender':
                if in_own_half(v):
                    res = self.tag_mgr.request_tag(v.name)
                    if res.get("tagged"):
                        self.flag_mgr.reset_flags_by_owner(res["tagged"])
            else:
                if in_own_half(v) and not v.carrying:
                    res = self.tag_mgr.request_tag(v.name)
                    if res.get("tagged"):
                        self.flag_mgr.reset_flags_by_owner(res["tagged"])

        # Attempts to grab flags (attackers will be in range on enemy side)
        for v in self.vehicles.values():
            self.flag_mgr.request_grab(v.name)

        # Scoring by home circle
        for v in self.vehicles.values():
            home_center = self.flags["blue_flag"] if v.team == "blue" else self.flags["red_flag"]
            delivered = self.flag_mgr.scored_goal(v.name, home_center, self.home_radius)
            if delivered:
                pass  # bus event already published

    def step(self):
        for name in self.vehicles:
            self._update_motion(name)
        self._update_managers()
        self.tag_mgr.tick()
        self.bus.drain()

# ------------------------------- Visualization
class Viewer:
    def __init__(self, world: World):
        self.W = world
        self.fig, self.ax = plt.subplots()
        self.vehicle_patches: Dict[str, patches.Polygon] = {}
        self.text = self.ax.text(0.5, -0.05, "", transform=self.ax.transAxes, ha="center", va="top")

        self.ax.set_xlim(0, self.W.w)
        self.ax.set_ylim(0, self.W.h)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Aquaticus Capture-the-Flag (minimal sim)")

        bx = [0, self.W.w/2, self.W.w/2, 0, 0]
        by = [0, 0, self.W.h, self.W.h, 0]
        rx = [self.W.w/2, self.W.w, self.W.w, self.W.w/2, self.W.w/2]  # FIXED (no self.w)
        ry = [0,          0,         self.W.h, self.W.h,   0]
        self.ax.plot(bx, by, linewidth=1, color='blue', alpha=0.2)
        self.ax.plot(rx, ry, linewidth=1, color='red', alpha=0.2)

        # Flags
        self.blue_flag_plot = self.ax.plot([self.W.flags["blue_flag"].x], [self.W.flags["blue_flag"].y], marker="o", color="blue")[0]
        self.red_flag_plot  = self.ax.plot([self.W.flags["red_flag"].x],  [self.W.flags["red_flag"].y],  marker="o", color="red")[0]

        # Home scoring circles (visualize 10 m diameter)
        self.ax.add_patch(plt.Circle((self.W.flags["blue_flag"].x, self.W.flags["blue_flag"].y),
                                     self.W.home_radius, fill=False, linestyle=":", color="blue", alpha=0.5))
        self.ax.add_patch(plt.Circle((self.W.flags["red_flag"].x, self.W.flags["red_flag"].y),
                                     self.W.home_radius, fill=False, linestyle=":", color="red", alpha=0.5))

        size = 3.0
        self.base_tri = [(size, 0), (-size/2, size * math.sqrt(3)/2), (-size/2, -size * math.sqrt(3)/2)]  # heron
        self.base_sq  = [(-size/2, -size/2), (size/2, -size/2), (size/2, size/2), (-size/2, size/2)]       # mokai

        for name, v in self.W.vehicles.items():
            color = 'blue' if v.team == "blue" else 'red'
            patch = patches.Polygon([[0,0]], fc=color, ec='black')
            self.ax.add_patch(patch)
            self.vehicle_patches[name] = patch

        self.carry_ring: Dict[str, plt.Circle] = {}
        self.tag_ring: Dict[str, plt.Circle] = {}
        for name in self.W.vehicles:
            cr = plt.Circle((0,0), 4.0, fill=False, color='green')
            tr = plt.Circle((0,0), 5.0, fill=False, linestyle="--", color='black')
            self.ax.add_patch(cr)
            self.ax.add_patch(tr)
            cr.set_visible(False)
            tr.set_visible(False)
            self.carry_ring[name] = cr
            self.tag_ring[name] = tr

        self.fig.tight_layout()

    def _update_frame(self, frame):
        self.W.step()

        for name, v in self.W.vehicles.items():
            heading = self.W.heading[name]
            cos_h, sin_h = math.cos(heading), math.sin(heading)
            base = self.base_tri if v.vtype == "heron" else self.base_sq
            rotated = [(px * cos_h - py * sin_h + v.pose.x, px * sin_h + py * cos_h + v.pose.y) for px, py in base]
            self.vehicle_patches[name].set_xy(rotated)

            cr = self.carry_ring[name]
            cr.center = (v.pose.x, v.pose.y)
            cr.set_visible(len(v.carrying) > 0)

            tr = self.tag_ring[name]
            tr.center = (v.pose.x, v.pose.y)
            tr.set_visible(now() < v.tagged_until)

        # Update flag markers (follow carrier if taken)
        for label, F in self.W.flag_mgr.flags.items():
            if F.owner is None:
                if label == "blue_flag":
                    self.blue_flag_plot.set_data([self.W.flags["blue_flag"].x], [self.W.flags["blue_flag"].y])
                else:
                    self.red_flag_plot.set_data([self.W.flags["red_flag"].x], [self.W.flags["red_flag"].y])
            else:
                v = self.W.vehicles[F.owner]
                if label == "blue_flag":
                    self.blue_flag_plot.set_data([v.pose.x], [v.pose.y])
                else:
                    self.red_flag_plot.set_data([v.pose.x], [v.pose.y])

        # Simple “scoreboard” placeholder: counts vehicles not carrying per team
        blue_idle = sum(1 for _ in self.W.vehicles.values() if _.team == "blue" and not _.carrying)
        red_idle  = sum(1 for _ in self.W.vehicles.values() if _.team == "red"  and not _.carrying)
        self.text.set_text(f"timestep={frame}  Blue idle:{blue_idle}  Red idle:{red_idle}  (rings: dashed=tagged, solid=carrying)")

        return list(self.vehicle_patches.values())

    def run(self, steps=300, interval_ms=60):
        ani = animation.FuncAnimation(self.fig, self._update_frame, frames=steps, interval=interval_ms, blit=False, repeat=False)
        plt.show()

# ------------------------------- Run it
def main():
    random.seed(2)
    world = World()
    viewer = Viewer(world)
    viewer.run(steps=400, interval_ms=60)

if __name__ == "__main__":
    main()
