# aquaticus_ctf.py
# Capture-the-Flag mini-sim aligned with Aquaticus rules:
# - 160x80m field, flags 20m from ends, 10m home zones
# - 2 humans (MOKAI) + 2 robots (Heron) per team
# - Triangles = robots, Squares (diamonds) = humans; all face heading
# - Tag: source in home half, target in enemy half, nearest within range, cooldown; OOB => tagged; home zone => clears tag
# - Flag: grab in 10m zone if not tagged; safe return scores

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, DefaultDict
from collections import defaultdict, deque
import math, random, time, threading

# =========================== Core data ===========================
@dataclass
class Pose:
    x: float
    y: float

@dataclass
class Vehicle:
    name: str
    team: str        # "blue" or "red"
    vtype: str       # "mokai" (human) or "heron" (robot)
    pose: Pose
    tagged_until: float = 0.0
    last_tag_time: float = -1e9
    carrying: List[str] = field(default_factory=list)

def now() -> float: return time.time()

def dist(a: Pose, b: Pose) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def point_in_convex_polygon(p: Pose, poly: List[Tuple[float,float]]) -> bool:
    n = len(poly)
    if n < 3: return False
    prev = None
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        cross = (x2-x1)*(p.y-y1) - (y2-y1)*(p.x-x1)
        if prev is None: prev = cross
        elif cross*prev < 0: return False
    return True

# =========================== Managers ===========================
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
        if name in self.vehicles:
            v = self.vehicles[name]; v.pose = pose; v.team = team; v.vtype = vtype
        else:
            self.vehicles[name] = Vehicle(name, team, vtype, pose)

    def _zone_for_team(self, team: str):  # halves for tag checks
        return self.cfg.zone_one if team == self.cfg.team_one else self.cfg.zone_two

    def _in_own_half(self, v: Vehicle) -> bool:
        return point_in_convex_polygon(v.pose, self._zone_for_team(v.team))

    def _cooldown_ready(self, v: Vehicle) -> bool:
        return now() - v.last_tag_time >= self.cfg.tag_min_interval

    def _eligible_targets(self, src: Vehicle) -> List[Vehicle]:
        opp = self.cfg.team_two if src.team == self.cfg.team_one else self.cfg.team_one
        out = []
        for v in self.vehicles.values():
            if v.team != opp: continue
            if now() < v.tagged_until: continue
            # target must be on enemy side (their enemy side == src's side)
            if not point_in_convex_polygon(v.pose, self._zone_for_team(src.team)): continue
            if dist(src.pose, v.pose) <= self.cfg.tag_range:
                out.append(v)
        return out

    def request_tag(self, src_name: str) -> Dict:
        self._event_id += 1
        if src_name not in self.vehicles:
            return {"event": self._event_id, "result":"reject", "reason":"unknown_src"}
        src = self.vehicles[src_name]
        if not self._in_own_half(src):  # must be on own side
            return {"event": self._event_id, "result":"reject", "reason":"zone"}
        if not self._cooldown_ready(src):  # cooldown
            return {"event": self._event_id, "result":"reject", "reason":"freq"}
        if now() < src.tagged_until:      # cannot tag while tagged
            return {"event": self._event_id, "result":"reject", "reason":"tagged"}

        cands = self._eligible_targets(src)
        if not cands:
            return {"event": self._event_id, "result":"ok", "tagged":None}
        tgt = min(cands, key=lambda t: dist(src.pose, t.pose))
        tgt.tagged_until = now() + self.cfg.tag_duration
        src.last_tag_time = now()
        return {"event": self._event_id, "result":"ok", "tagged":tgt.name, "expires_at":tgt.tagged_until}

    def clear_tag_if_home(self, v: Vehicle, home_center: Tuple[float,float], home_radius: float):
        # If a tagged vehicle returns to home zone, clear immediately
        if now() < v.tagged_until:
            if dist(v.pose, Pose(*home_center)) <= home_radius:
                v.tagged_until = 0.0

    def tick(self) -> None:
        """Optional housekeeping. Tags expire via timestamps, so this can be a no-op."""
        # If you want to actively clean anything, do it here. For example:
        # now_ts = now()
        # for v in self.vehicles.values():
        #     if v.tagged_until < now_ts:
        #         v.tagged_until = 0.0
        pass

@dataclass
class Flag:
    label: str
    pose: Pose
    grab_range: float = 10.0
    owner: Optional[str] = None

class FlagManager:
    def __init__(self):
        self.flags: Dict[str, Flag] = {}
        self.vehicles: Dict[str, Vehicle] = {}

    def add_flag(self, label: str, x: float, y: float, grab_range: float = 10.0):
        self.flags[label] = Flag(label, Pose(x, y), grab_range)

    def update_node_report(self, name: str, team: str, vtype: str, pose: Pose):
        if name in self.vehicles:
            v = self.vehicles[name]; v.pose = pose; v.team = team; v.vtype = vtype
        else:
            self.vehicles[name] = Vehicle(name, team, vtype, pose)

    def request_grab(self, vname: str) -> Dict:
        if vname not in self.vehicles:
            return {"result":"deny", "reason":"unknown_src"}
        v = self.vehicles[vname]
        if now() < v.tagged_until:
            return {"result":"deny", "reason":"tagged"}

        grabbed = []
        for f in self.flags.values():
            if f.owner is None and dist(v.pose, f.pose) <= f.grab_range:
                f.owner = vname
                v.carrying.append(f.label)
                grabbed.append(f.label)
        return {"result":"ok", "grabbed":grabbed} if grabbed else {"result":"deny", "reason":"nothing_in_range"}

    def reset_flag_by_label(self, label: str):
        if label in self.flags:
            owner = self.flags[label].owner
            if owner and owner in self.vehicles:
                try: self.vehicles[owner].carrying.remove(label)
                except ValueError: pass
            self.flags[label].owner = None

    def reset_flags_by_owner(self, vname: str):
        for f in self.flags.values():
            if f.owner == vname: f.owner = None
        if vname in self.vehicles: self.vehicles[vname].carrying.clear()

    def scored_goal(self, vname: str, home_center: Tuple[float,float], home_radius: float) -> Optional[List[str]]:
        if vname not in self.vehicles: return None
        v = self.vehicles[vname]
        if now() < v.tagged_until: self.reset_flags_by_owner(vname); return None
        if dist(v.pose, Pose(*home_center)) > home_radius: return None
        delivered = list(v.carrying)
        for lbl in delivered: self.reset_flag_by_label(lbl)
        v.carrying.clear()
        return delivered

# =========================== Tiny async bus ===========================
class AsyncBus:
    def __init__(self):
        self._subs: DefaultDict[str, List[Callable[[dict], None]]] = defaultdict(list)
        self._queue: "deque[Tuple[str,dict]]" = deque()
        self._lock = threading.Lock()
    def subscribe(self, topic: str, cb: Callable[[dict], None]): self._subs[topic].append(cb)
    def publish(self, topic: str, payload: dict):
        with self._lock: self._queue.append((topic, payload))
    def drain(self, max_msgs: int = 1000):
        processed = 0
        while processed < max_msgs:
            with self._lock:
                if not self._queue: break
                topic, payload = self._queue.popleft()
            for cb in self._subs.get(topic, []): cb(payload)
            processed += 1

# =========================== World & Logic ===========================
@dataclass
class AgentCtrl:
    max_speed: float      # m/s
    wander: float = 0.12  # rad/s jitter
    goal_gain: float = 1.2
    wall_gain: float = 2.2
    wall_pad: float = 6.0

class World:
    def __init__(self, width=160.0, height=80.0, dt=0.4):
        self.w, self.h, self.dt = width, height, dt
        self.bus = AsyncBus()

        # Tag halves for rule checks
        self.blue_half = [(0,0),(self.w/2,0),(self.w/2,self.h),(0,self.h)]
        self.red_half  = [(self.w/2,0),(self.w,0),(self.w,self.h),(self.w/2,self.h)]

        self.tag_mgr = TagManager(TagManagerConfig(
            zone_one=self.blue_half, zone_two=self.red_half
        ))
        self.flag_mgr = FlagManager()

        # Flags & 10m home zones
        self.blue_flag = Pose(20, self.h/2)
        self.red_flag  = Pose(self.w-20, self.h/2)
        self.home_radius = 10.0
        self.flag_mgr.add_flag("blue_flag", self.blue_flag.x, self.blue_flag.y, 10.0)
        self.flag_mgr.add_flag("red_flag",  self.red_flag.x,  self.red_flag.y,  10.0)

        # Vehicles: 2 humans (mokai) + 2 robots (heron) per team
        self.vehicles: Dict[str, Vehicle] = {}
        self._spawn_team("blue", x=self.w*0.22, ymid=self.h*0.5)
        self._spawn_team("red",  x=self.w*0.78, ymid=self.h*0.5)

        # Controllers & headings (humans faster per spec ~4 m/s; robots ~1.8 m/s)
        self.ctrl: Dict[str, AgentCtrl] = {}
        self.heading: Dict[str, float] = {}
        for n, v in self.vehicles.items():
            max_spd = 1.8 if v.vtype=="heron" else 4.0
            self.ctrl[n] = AgentCtrl(max_speed=max_spd)
            self.heading[n] = 0.0 if v.team=="blue" else math.pi

        self.score = {"blue":0, "red":0}

    def _spawn_team(self, team: str, x: float, ymid: float):
        # Order: two robots (heron) + two humans (mokai)
        layout = [("heron",-18), ("heron",18), ("mokai",-6), ("mokai",6)]
        for i,(vtype,dy) in enumerate(layout, start=1):
            name = f"{team[0]}{i}"
            v = Vehicle(name=name, team=team, vtype=vtype, pose=Pose(x, ymid+dy))
            self.vehicles[name] = v
            self.tag_mgr.update_node_report(name, team, vtype, v.pose)
            self.flag_mgr.update_node_report(name, team, vtype, v.pose)

    def _goal_for(self, v: Vehicle) -> Pose:
        enemy_flag = self.red_flag if v.team=="blue" else self.blue_flag
        home_flag  = self.blue_flag if v.team=="blue" else self.red_flag
        return home_flag if v.carrying else enemy_flag

    def _own_half(self, team: str): return self.blue_half if team=="blue" else self.red_half
    def _home_center(self, team: str):
        f = self.blue_flag if team=="blue" else self.red_flag
        return (f.x, f.y)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a <= -math.pi: a += 2*math.pi
        while a >  math.pi:  a -= 2*math.pi
        return a

    def _wall_steer(self, v: Vehicle) -> float:
        pad = self.ctrl[v.name].wall_pad; gain = self.ctrl[v.name].wall_gain
        steer = 0.0
        h = self.heading[v.name]
        if v.pose.x < pad: steer += gain * self._wrap_angle(0 - h)
        if self.w - v.pose.x < pad: steer += gain * self._wrap_angle(math.pi - h)
        if v.pose.y < pad: steer += gain * self._wrap_angle(math.pi/2 - h)
        if self.h - v.pose.y < pad: steer += gain * self._wrap_angle(-math.pi/2 - h)
        return steer

    def _update_motion(self, name: str):
        v = self.vehicles[name]; c = self.ctrl[name]
        # Out-of-bounds => auto-tag and nudge inside
        if not (0 <= v.pose.x <= self.w and 0 <= v.pose.y <= self.h):
            v.tagged_until = max(v.tagged_until, now() + 5.0)
            v.pose.x = min(max(v.pose.x, 0), self.w); v.pose.y = min(max(v.pose.y, 0), self.h)

        tgt = self._goal_for(v)
        desired = math.atan2(tgt.y - v.pose.y, tgt.x - v.pose.x)
        steer = c.goal_gain * self._wrap_angle(desired - self.heading[name]) + self._wall_steer(v)
        jitter = random.uniform(-c.wander, c.wander)
        self.heading[name] = self._wrap_angle(self.heading[name] + 0.5*(steer)*self.dt + jitter*self.dt)

        speed = c.max_speed * (0.5 + 0.5*random.random())
        if now() < v.tagged_until:
            speed *= 0.35  # crawl when tagged
            # drift toward home when tagged
            hc = self._home_center(v.team)
            desired_home = math.atan2(hc[1] - v.pose.y, hc[0] - v.pose.x)
            self.heading[name] = self._wrap_angle(self.heading[name] + 0.6*self._wrap_angle(desired_home - self.heading[name])*self.dt)

        v.pose.x = min(max(v.pose.x + math.cos(self.heading[name])*speed*self.dt, 0), self.w)
        v.pose.y = min(max(v.pose.y + math.sin(self.heading[name])*speed*self.dt, 0), self.h)

    def _update_managers(self):
        # node reports
        for v in self.vehicles.values():
            self.tag_mgr.update_node_report(v.name, v.team, v.vtype, v.pose)
            self.flag_mgr.update_node_report(v.name, v.team, v.vtype, v.pose)

        # try tags if in own half
        for v in self.vehicles.values():
            if point_in_convex_polygon(v.pose, self._own_half(v.team)):
                res = self.tag_mgr.request_tag(v.name)
                if res.get("tagged"):
                    self.flag_mgr.reset_flags_by_owner(res["tagged"])

        # grab attempts near flags
        for v in self.vehicles.values():
            self.flag_mgr.request_grab(v.name)

        # clear tags if inside 10m home zone; score if carrying there
        for v in self.vehicles.values():
            hc = self._home_center(v.team)
            self.tag_mgr.clear_tag_if_home(v, hc, self.home_radius)
            delivered = self.flag_mgr.scored_goal(v.name, hc, self.home_radius)
            if delivered:
                self.score[v.team] += len(delivered)

    def step(self):
        for name in self.vehicles: self._update_motion(name)
        self._update_managers()
        self.tag_mgr.tick(); self.bus.drain()

# =========================== Visualization ===========================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon, Circle

class Viewer:
    def __init__(self, world: World):
        self.W = world
        self.fig, self.ax = plt.subplots(figsize=(10,5))
        self.fig.subplots_adjust(top=0.9)
        self.fig.suptitle("Aquaticus Capture‑the‑Flag (2 humans + 2 robots per team)", y=0.98)

        self.ax.set_xlim(0, self.W.w); self.ax.set_ylim(0, self.W.h)
        self.ax.set_aspect("equal", "box")

        # field midline and boundary
        self.ax.axvline(self.W.w/2, lw=2, alpha=0.5)
        self.ax.plot([0, self.W.w, self.W.w, 0, 0], [0,0,self.W.h,self.W.h,0], lw=1)

        # home zones (10m circles)
        self.ax.add_patch(Circle((self.W.blue_flag.x, self.W.blue_flag.y), self.W.home_radius, fill=False, ls="--"))
        self.ax.add_patch(Circle((self.W.red_flag.x,  self.W.red_flag.y),  self.W.home_radius, fill=False, ls="--"))

        # flags
        self.blue_flag_plot = self.ax.plot(self.W.blue_flag.x, self.W.blue_flag.y, marker="o", ms=8)[0]
        self.red_flag_plot  = self.ax.plot(self.W.red_flag.x,  self.W.red_flag.y,  marker="o", ms=8)[0]

        # vehicle glyphs: robots=triangles, humans=diamonds (square rotated)
        self.glyphs: Dict[str, RegularPolygon] = {}
        self.carry_ring: Dict[str, Circle] = {}
        self.tag_ring: Dict[str, Circle] = {}

        for n, v in self.W.vehicles.items():
            if v.vtype == "heron":
                poly = RegularPolygon((v.pose.x, v.pose.y), numVertices=3, radius=2.8,
                                      orientation=self.W.heading[n],
                                      facecolor=("tab:blue" if v.team=="blue" else "tab:red"),
                                      edgecolor="k", lw=0.8)
            else:
                poly = RegularPolygon((v.pose.x, v.pose.y), numVertices=4, radius=2.6,
                                      orientation=self.W.heading[n] + math.pi/4.0,  # diamond shape
                                      facecolor=("lightskyblue" if v.team=="blue" else "lightcoral"),
                                      edgecolor="k", lw=0.8)
            self.ax.add_patch(poly)
            self.glyphs[n] = poly

            cr = Circle((v.pose.x, v.pose.y), 2.5, fill=False, lw=1.5)  # carrying indicator
            tr = Circle((v.pose.x, v.pose.y), 3.6, fill=False, lw=1.2, ls="--")  # tagged indicator
            self.ax.add_patch(cr); self.ax.add_patch(tr)
            cr.set_visible(False); tr.set_visible(False)
            self.carry_ring[n] = cr; self.tag_ring[n] = tr

        # HUD top-right
        self.hud = self.ax.text(0.995, 0.995, "", transform=self.ax.transAxes,
                                ha="right", va="top",
                                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))

    def _update_frame(self, frame):
        self.W.step()

        for n, v in self.W.vehicles.items():
            g = self.glyphs[n]
            g.set_center((v.pose.x, v.pose.y))
            # triangles face heading; diamonds are +pi/4 offset
            if v.vtype == "heron":
                g.orientation = self.W.heading[n]
            else:
                g.orientation = self.W.heading[n] + math.pi/4.0

            cr = self.carry_ring[n]; tr = self.tag_ring[n]
            cr.center = (v.pose.x, v.pose.y); tr.center = (v.pose.x, v.pose.y)
            cr.set_visible(len(v.carrying) > 0)
            tr.set_visible(now() < v.tagged_until)

        # flags follow owner else stay home
        for label, F in self.W.flag_mgr.flags.items():
            if F.owner is None:
                if label == "blue_flag":
                    self.blue_flag_plot.set_data(self.W.blue_flag.x, self.W.blue_flag.y)
                else:
                    self.red_flag_plot.set_data(self.W.red_flag.x, self.W.red_flag.y)
            else:
                v = self.W.vehicles[F.owner]
                (self.blue_flag_plot if label=="blue_flag" else self.red_flag_plot).set_data(v.pose.x, v.pose.y)

        self.hud.set_text(
            f"timestep: {frame}\n"
            f"score  blue:{self.W.score['blue']}  red:{self.W.score['red']}\n"
            f"dashed = tagged, solid = carrying"
        )
        return list(self.glyphs.values())

    def run(self, steps=600, interval_ms=60):
        ani = animation.FuncAnimation(self.fig, self._update_frame,
                                      frames=steps, interval=interval_ms,
                                      blit=False, repeat=False)
        plt.show()

# =========================== Run ===========================
def main():
    random.seed(4)
    world = World()
    viewer = Viewer(world)
    viewer.run(steps=600, interval_ms=60)

if __name__ == "__main__":
    main()
