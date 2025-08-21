import pygame
from queue import PriorityQueue
import math, random, os, json

# ===== Debug toggles =====
DEBUG_TICKS = 200         # how many ticks to print (0 = off)
DEBUG_PER_TICK = True     # print every tick up to DEBUG_TICKS
DEBUG_CONFLICTS = True    # print resolution conflicts
random.seed(7)

# ===== World =====
ROWS = 20
COLS = 20
CELL_SIZE = 30
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

WHITE = (255,255,255); BLACK=(0,0,0); GREY=(128,128,128); BLUE=(0,0,255); RED=(255,0,0)

TICKS_PER_SECOND = 5
STUN_TICKS    = 4
RESPAWN_TICKS = 6

# ===== Influence / congestion tuning =====
ENEMY_REPEL_NEAR   = 2.25   # was 1.5
ALLY_ATTRACT       = 0.12
CENTER_COL_PENALTY = 0.6    # stronger mid-choke penalty
CONGESTION_GAIN    = 0.35   # per conflict
CONGESTION_DECAY   = 0.92   # per tick (exponential decay)
NO_BACKTRACK       = True   # avoid immediate backtrack if any other option exists
SIDE_STEP_STUCK_N  = 2      # if stuck N ticks, prefer lateral alternative
YIELD_COOLDOWN_T   = 2

# ===== Utilities =====
def h(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return abs(x1-x2) + abs(y1-y2)

def get_neighbors(grid, r, c):
    out=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<ROWS and 0<=nc<COLS and grid[nr][nc] != 1:
            out.append((nr,nc))
    return out

# ===== A* variants =====
def a_star_influenced(grid, start, goal, infl):
    if start == goal: return []
    openq = PriorityQueue(); count = 0
    g = {start: 0.0}
    came = {}
    openq.put((h(start,goal), count, start))
    closed=set()
    while not openq.empty():
        f,_,cur = openq.get()
        if cur in closed: continue
        closed.add(cur)
        if cur == goal:
            path=[]
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        for nb in get_neighbors(grid, *cur):
            # base step cost 1, add bounded influence (can be negative but keep >0)
            step_cost = 1.0 + max(-0.9, infl[nb[0]][nb[1]])
            ng = g[cur] + step_cost
            if ng < g.get(nb, 1e9):
                g[nb] = ng
                came[nb] = cur
                count += 1
                openq.put((ng + h(nb,goal), count, nb))
    return None

def build_influence(grid, all_agents, team, congestion):
    INF = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]
    # enemy repulsion, ally slight attraction
    for a in all_agents:
        sgn = -ENEMY_REPEL_NEAR if a.team != team else +ALLY_ATTRACT
        ar,ac = a.pos
        r0 = max(0,ar-6); r1 = min(ROWS,ar+7)
        c0 = max(0,ac-6); c1 = min(COLS,ac+7)
        for r in range(r0, r1):
            for c in range(c0, c1):
                d = abs(r-ar) + abs(c-ac)
                if d==0:
                    val = -3.0*ENEMY_REPEL_NEAR if sgn<0 else 0.5*ALLY_ATTRACT
                else:
                    val = sgn * (1.0/d)
                INF[r][c] += val
    # discourage center wall column to promote flanks
    mid = COLS//2
    for r in range(ROWS):
        INF[r][mid] -= CENTER_COL_PENALTY
    # add congestion (learned from recent conflicts)
    for r in range(ROWS):
        for c in range(COLS):
            INF[r][c] += congestion[r][c]
    return INF

# ===== Entities =====
class Flag:
    def __init__(self, team, home_r, home_c):
        self.team = team
        self.home_pos = (home_r, home_c)
        self.pos = self.home_pos
        self.carried_by = None
    def get_pos(self):
        return self.carried_by.pos if self.carried_by else self.pos

# expanded tactics for more independent behaviors, including hunt (attack) and evade-focused
TACTICS = ["left_gap","right_gap","center","shadow","home_intercept","hunt","evade_flank","support_ally"]

class StrategyLearner:
    def __init__(self, id, alpha=0.25, epsilon=0.20, gamma=0.9, path="policy_{}.json"):
        self.id = id
        self.alpha=alpha; self.eps=epsilon; self.gamma=gamma
        self.path = path.format(id)
        self.Q = {t: 0.0 for t in TACTICS}
        if os.path.exists(self.path):
            try: self.Q.update(json.load(open(self.path)))
            except: pass
        self.last_choice=None
        self.episode_reward=0.0

    def choose(self):
        if random.random() < self.eps:
            self.last_choice = random.choice(TACTICS)
        else:
            self.last_choice = max(TACTICS, key=lambda t: self.Q.get(t,0.0))
        return self.last_choice

    def add_reward(self, r):
        self.episode_reward += r

    def end_episode(self, did_win):
        R = self.episode_reward + (1.0 if did_win else -1.0)
        t = self.last_choice or max(TACTICS, key=lambda x: self.Q.get(x,0.0))
        self.Q[t] = (1-self.alpha)*self.Q.get(t,0.0) + self.alpha*R
        try: json.dump(self.Q, open(self.path,"w"))
        except: pass
        self.episode_reward=0.0; self.last_choice=None

class Agent:
    def __init__(self, r,c, team, role='auto'):
        self.pos = (r,c)
        self.spawn_pos = (r,c)
        self.team = team
        self.role = role              # 'attacker'|'defender'|'auto'
        self.carrying_flag = False
        self.facing = 90 if team=='blue' else 270
        self.stun_ticks = 0
        self.respawn_ticks = 0
        self.current_target_pos = None
        self.path = []
        # anti-stuck
        self.stuck_count = 0
        self.prev_pos = self.pos
        self.yield_cooldown = 0
        self.patrol_phase = 0  # defenders tiny patrol
        # independent learner
        self.brain = StrategyLearner(team + '_' + role)
        self.tactic = self.brain.choose()

    @property
    def alive(self): return self.respawn_ticks <= 0
    @property
    def stunned(self): return self.stun_ticks > 0

    def choose_role(self, blue_flag, red_flag):
        if self.role != 'auto': return self.role
        own_flag = blue_flag if self.team=='blue' else red_flag
        if own_flag.carried_by and own_flag.carried_by.team != self.team:
            return 'defender'
        # default by spawn proximity to edge
        if self.team=='blue':
            return 'defender' if self.spawn_pos[1] <= 1 else 'attacker'
        else:
            return 'defender' if self.spawn_pos[1] >= COLS-2 else 'attacker'

    def target_for_role(self, blue_flag, red_flag, all_agents):
        enemy_flag = red_flag if self.team=='blue' else blue_flag
        own_flag   = blue_flag if self.team=='blue' else red_flag
        home_pos   = blue_flag.home_pos if self.team=='blue' else red_flag.home_pos

        if self.carrying_flag:
            return home_pos

        role = self.choose_role(blue_flag, red_flag)
        tactic = self.tactic

        if role == 'attacker':
            er,ec = enemy_flag.get_pos()
            # tactic waypoints
            if tactic == "left_gap":   wp = (er, max(0, COLS//2 - 3))
            elif tactic == "right_gap":wp = (er, min(COLS-1, COLS//2 + 3))
            elif tactic == "center":   wp = (er, COLS//2)
            elif tactic == "shadow":
                allies = [a for a in all_agents if a.team==self.team and a is not self and a.alive]
                if allies:
                    ally = min(allies, key=lambda x: h(self.pos, x.pos))
                    ar,ac = ally.pos
                    dr = 1 if ar<er else -1 if ar>er else 0
                    dc = 1 if ac<ec else -1 if ac>ec else 0
                    wp = (max(0,min(ROWS-1, ar - dr)), max(0,min(COLS-1, ac - dc)))
                else:
                    wp = enemy_flag.get_pos()
            elif tactic == "home_intercept":
                enemies = [a for a in all_agents if a.team != self.team and a.alive]
                if enemies:
                    e = min(enemies, key=lambda x: h(self.pos,x.pos) + (0 if in_home_for(self.team, x.pos) else 4))
                    return e.pos
                wp = enemy_flag.get_pos()
            elif tactic == "hunt":
                enemies = [a for a in all_agents if a.team != self.team and a.alive]
                if enemies:
                    return min(enemies, key=lambda x: h(self.pos, x.pos)).pos
                wp = enemy_flag.get_pos()
            elif tactic == "evade_flank":
                # aim for a flank with higher repel (evade built into influence)
                wp = (er, random.choice([max(0, COLS//2 - 4), min(COLS-1, COLS//2 + 4)]))
            elif tactic == "support_ally":
                allies = [a for a in all_agents if a.team==self.team and a is not self and a.alive and a.carrying_flag]
                if allies:
                    ally = min(allies, key=lambda x: h(self.pos, x.pos))
                    return ally.pos
                wp = enemy_flag.get_pos()
            else:
                wp = enemy_flag.get_pos()
            # go via waypoint if meaningful
            return wp if h(self.pos, wp) > 1 else enemy_flag.get_pos()

        # defender: chase intruders in home, else small patrol around flag
        enemies = [a for a in all_agents if a.team != self.team and a.alive]
        home_enemies=[]
        for e in enemies:
            ec = e.pos[1]
            in_our_home = (self.team=='blue' and ec < COLS//2) or (self.team=='red' and ec >= COLS//2)
            if in_our_home: home_enemies.append(e)
        if home_enemies:
            e = min(home_enemies, key=lambda x: h(self.pos, x.pos))
            return e.pos

        fr,fc = own_flag.home_pos
        ring=[(fr,max(0,fc-1)),(fr,min(COLS-1,fc+1)),(max(0,fr-1),fc),(min(ROWS-1,fr+1),fc)]
        # tiny patrol: rotate target every few ticks to avoid yo-yo on one cell
        if self.patrol_phase % 8 == 0:
            self.patrol_choice = random.choice(ring)
        self.patrol_phase += 1
        return self.patrol_choice

    def plan_next_step(self, grid, blue_flag, red_flag, all_agents, congestion):
        if not self.alive or self.stunned:
            self.current_target_pos = None
            self.path = []
            return self.pos

        self.current_target_pos = self.target_for_role(blue_flag, red_flag, all_agents)
        infl = build_influence(grid, all_agents, self.team, congestion)
        path = a_star_influenced(grid, self.pos, self.current_target_pos, infl)

        if not path or len(path)==0:
            # fallback: greedy neighbor toward target if any
            nbs = get_neighbors(grid, *self.pos)
            if nbs:
                nbs.sort(key=lambda p: h(p, self.current_target_pos))
                nxt = nbs[0]
            else:
                nxt = self.pos
        else:
            nxt = path[0]
            self.path = path

        # no immediate backtrack if option exists
        if NO_BACKTRACK and self.prev_pos and nxt == self.prev_pos:
            nbs = get_neighbors(grid, *self.pos)
            alts = [p for p in nbs if p != self.prev_pos]
            if alts:
                alts.sort(key=lambda p: h(p, self.current_target_pos))
                nxt = alts[0]

        # anti-stickiness: if stuck, try lateral best neighbor not equal to straight-line
        if self.stuck_count >= SIDE_STEP_STUCK_N:
            nbs = get_neighbors(grid, *self.pos)
            alt = [p for p in nbs if p != nxt]
            if alt:
                alt.sort(key=lambda p: h(p, self.current_target_pos))
                nxt = alt[0]

        # update facing
        dr = nxt[0]-self.pos[0]; dc = nxt[1]-self.pos[1]
        if dr==-1: self.facing=0
        elif dr==1: self.facing=180
        elif dc==1: self.facing=90
        elif dc==-1:self.facing=270
        return nxt

    def apply_move(self, next_pos):
        if self.alive and not self.stunned:
            moved = (next_pos != self.pos)
            self.prev_pos = self.pos
            self.pos = next_pos
            if moved and self.path: self.path.pop(0)
            self.stuck_count = 0 if moved else (self.stuck_count + 1)

    def tick_timers(self):
        if self.stun_ticks>0: self.stun_ticks -= 1
        if self.respawn_ticks>0:
            self.respawn_ticks -= 1
            if self.respawn_ticks == 0:
                self.pos = self.spawn_pos
                self.facing = 90 if self.team=='blue' else 270
                self.carrying_flag=False
                self.path=[]; self.current_target_pos=None

# ===== Rendering =====
def draw(win, grid, blue_agents, red_agents, blue_flag, red_flag):
    win.fill(WHITE)
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c]==1:
                pygame.draw.rect(win, BLACK, (c*CELL_SIZE,r*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    pygame.draw.line(win,(200,200,200),(COLS//2*CELL_SIZE,0),(COLS//2*CELL_SIZE,HEIGHT),2)
    bx = blue_flag.home_pos[1]*CELL_SIZE+CELL_SIZE//2
    by = blue_flag.home_pos[0]*CELL_SIZE+CELL_SIZE//2
    rx = red_flag.home_pos[1]*CELL_SIZE+CELL_SIZE//2
    ry = red_flag.home_pos[0]*CELL_SIZE+CELL_SIZE//2
    pygame.draw.circle(win, BLUE,(bx,by),CELL_SIZE//2,2)
    pygame.draw.circle(win, RED,(rx,ry),CELL_SIZE//2,2)
    if not blue_flag.carried_by:
        pygame.draw.rect(win, BLUE,(blue_flag.pos[1]*CELL_SIZE, blue_flag.pos[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    if not red_flag.carried_by:
        pygame.draw.rect(win, RED,(red_flag.pos[1]*CELL_SIZE, red_flag.pos[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    all_agents = blue_agents+red_agents
    for a in all_agents:
        color = BLUE if a.team=='blue' else RED
        cx=a.pos[1]*CELL_SIZE+CELL_SIZE//2; cy=a.pos[0]*CELL_SIZE+CELL_SIZE//2
        r= CELL_SIZE//3; s3 = math.sqrt(3)/2
        base=[(cx,cy-r),(cx-r*s3,cy+r/2),(cx+r*s3,cy+r/2)]
        ang=math.radians(a.facing); ct,st = math.cos(ang), math.sin(ang)
        pts=[]
        for px,py in base:
            dx,dy=px-cx,py-cy
            pts.append((cx+dx*ct-dy*st, cy+dx*st+dy*ct))
        pygame.draw.polygon(win, color, pts)
        if a.stunned: pygame.draw.circle(win,(255,165,0),(cx,cy),CELL_SIZE//3,2)
        if not a.alive: pygame.draw.circle(win,(80,80,80),(cx,cy),CELL_SIZE//3,2)
        if a.carrying_flag:
            fcol = RED if a.team=='blue' else BLUE
            pygame.draw.rect(win,fcol,(cx-CELL_SIZE//4,cy-CELL_SIZE//4,CELL_SIZE//2,CELL_SIZE//2))
    for i in range(ROWS+1):
        pygame.draw.line(win, GREY, (0,i*CELL_SIZE),(WIDTH,i*CELL_SIZE))
    for j in range(COLS+1):
        pygame.draw.line(win, GREY, (j*CELL_SIZE,0),(j*CELL_SIZE,HEIGHT))
    pygame.display.update()

# ===== Helpers =====
def in_home_for(team, pos):
    c = pos[1]
    return (team=='blue' and c < COLS//2) or (team=='red' and c >= COLS//2)

def resolve_moves(all_agents, proposed, tick):
    """
    Improved arbitration:
      - Only ALIVE & NOT STUNNED agents participate in conflicts.
      - Resolve direct swaps first. Allies both pass through; enemies: winner moves, loser holds (and gets yield cooldown).
      - Then resolve remaining cell conflicts by priority.
      - Priority: flag carrier >> closer to target >> less stuck >> random; agents that recently yielded are throttled.
      - Returns (final_positions, conflict_cells) so caller can update congestion heatmap.
    """
    final_positions = {ag: ag.pos for ag in all_agents}
    reserved=set()
    conflict_cells=set()

    active = [a for a in all_agents if a.alive and not a.stunned]

    def is_swap(a,b):
        return proposed.get(b)==a.pos and proposed.get(a)==b.pos and proposed[a]!=a.pos and proposed[b]!=b.pos

    def prio(ag):
        carry = 10000 if ag.carrying_flag else 0
        tgt = getattr(ag,'current_target_pos', ag.pos) or ag.pos
        clos = -h(ag.pos,tgt)
        stuck_bonus = -getattr(ag, 'stuck_count', 0)
        yield_pen = -5 if getattr(ag,'yield_cooldown',0)>0 else 0
        return carry*2 + clos*10 + stuck_bonus + yield_pen + random.random()*0.01

    processed=set()

    # Pass 0: direct swaps among ACTIVE only
    for i,a in enumerate(active):
        if a in processed: continue
        for b in active[i+1:]:
            if b in processed: continue
            if is_swap(a,b):
                if a.team == b.team:
                    final_positions[a] = proposed[a]
                    final_positions[b] = proposed[b]
                    reserved.add(proposed[a]); reserved.add(proposed[b])
                    if DEBUG_CONFLICTS and DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                        print(f"[tick {tick}] ALLOW ALLY SWAP: {a.team}@{a.pos}<->{b.team}@{b.pos}")
                else:
                    winner = max([a,b], key=prio)
                    loser  = b if winner is a else a
                    final_positions[winner] = proposed[winner]
                    reserved.add(proposed[winner])
                    setattr(loser, 'yield_cooldown', YIELD_COOLDOWN_T)
                    if DEBUG_CONFLICTS and DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                        print(f"[tick {tick}] ENEMY SWAP WINNER: {winner.team}@{winner.pos}->{proposed[winner]} ; loser {loser.team} holds")
                processed.add(a); processed.add(b)
                break

    # Pass 1: remaining, group by target cell (ACTIVE only)
    remaining=[ag for ag in active if ag not in processed]
    cell_to_agents={}
    for ag in remaining:
        cell_to_agents.setdefault(proposed[ag],[]).append(ag)

    for cell, contenders in cell_to_agents.items():
        if cell in reserved:
            continue
        if len(contenders)==1:
            a = contenders[0]
            if getattr(a,'yield_cooldown',0) > 0:
                setattr(a,'yield_cooldown', getattr(a,'yield_cooldown')-1)
                # hold
                continue
            final_positions[a]=cell; reserved.add(cell)
        else:
            conflict_cells.add(cell)
            winner = max(contenders, key=prio)
            final_positions[winner]=cell; reserved.add(cell)
            if DEBUG_CONFLICTS and DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                losers=[ag for ag in contenders if ag is not winner]
                print(f"[tick {tick}] CONFLICT {cell}: winner {winner.team}@{winner.pos}->{cell}; losers hold {[(l.team,l.pos) for l in losers]}")

    return final_positions, conflict_cells

# ===== Pygame setup =====
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI CTF — independent learners + expanded tactics")

# Grid (0: open, 1: wall)
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
# middle barrier with gaps
for r in range(ROWS):
    if r % 5 != 0:
        grid[r][COLS//2] = 1

# Flags
blue_flag = Flag('blue', ROWS//2, 2)
red_flag  = Flag('red',  ROWS//2, COLS-3)

# Agents with independent brains
blue_agents = [
    Agent(ROWS//2 - 2, 1, 'blue', role='defender'),
    Agent(ROWS//2 + 2, 1, 'blue', role='attacker')
]
red_agents = [
    Agent(ROWS//2 - 2, COLS - 2, 'red', role='defender'),
    Agent(ROWS//2 + 2, COLS - 2, 'red', role='attacker')
]
all_agents = blue_agents + red_agents

# Congestion heatmap (decays over time), added to influence costs
congestion = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]

clock = pygame.time.Clock()
run=True
blue_score=0; red_score=0
tick=0

while run:
    clock.tick(TICKS_PER_SECOND)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    tick += 1
    # timers
    for ag in all_agents:
        ag.tick_timers()

    # PLAN
    proposed={}
    for ag in all_agents:
        nxt = ag.plan_next_step(grid, blue_flag, red_flag, all_agents, congestion)
        if (not ag.alive) or ag.stunned: nxt = ag.pos
        proposed[ag]=nxt

    # LOG plans
    if DEBUG_PER_TICK and tick<=DEBUG_TICKS:
        for ag in all_agents:
            tg = ag.current_target_pos
            path_len = len(ag.path) if ag.path else 0
            print(f"[tick {tick:03d}] {ag.team[:1].upper()}-{ag.role[:3]:<3} pos={ag.pos} "
                  f"-> prop={proposed[ag]} target={tg} path_len={path_len} "
                  f"{'CARRY' if ag.carrying_flag else ''}{' STUN' if ag.stunned else ''}{' DEAD' if not ag.alive else ''}")

    # RESOLVE (ACTIVE-ONLY) + collect conflict cells
    final_positions, conflict_cells = resolve_moves(all_agents, proposed, tick)

    # Apply congestion update (decay, then reinforce conflict cells)
    for r in range(ROWS):
        for c in range(COLS):
            congestion[r][c] *= CONGESTION_DECAY
    for (rr,cc) in conflict_cells:
        congestion[rr][cc] += CONGESTION_GAIN

    # APPLY
    for ag in all_agents:
        ag.apply_move(final_positions[ag])

    # FLAGS: pickup
    for ag in all_agents:
        if not ag.alive: continue
        enemy_flag = red_flag if ag.team=='blue' else blue_flag
        if enemy_flag.carried_by is None and ag.pos == enemy_flag.get_pos():
            enemy_flag.carried_by = ag
            enemy_flag.pos = ag.pos
            ag.carrying_flag = True
            if DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                print(f"[tick {tick}] {ag.team} picked up flag at {ag.pos}")
            # small shaped reward
            ag.brain.add_reward(0.2)

    # SCORE
    scored = False
    scorer = None
    scored_team = None
    for ag in all_agents:
        if ag.carrying_flag:
            home = blue_flag.home_pos if ag.team=='blue' else red_flag.home_pos
            if ag.pos == home:
                scored = True
                scorer = ag
                scored_team = ag.team
                if ag.team=='blue': blue_score+=1
                else: red_score+=1
                print(f"Scores: Blue {blue_score}, Red {red_score}")
                enemy_flag = red_flag if ag.team=='blue' else blue_flag
                enemy_flag.carried_by=None
                enemy_flag.pos = enemy_flag.home_pos
                ag.carrying_flag=False

    # COMBAT (adjacent)
    for ag in all_agents:
        if not ag.alive: continue
        enemies = red_agents if ag.team=='blue' else blue_agents
        for e in enemies:
            if not e.alive: continue
            if abs(e.pos[0]-ag.pos[0]) + abs(e.pos[1]-ag.pos[1]) <= 1:
                if in_home_for(ag.team, ag.pos):
                    # kill
                    if e.carrying_flag:
                        enemy_flag = blue_flag if e.team=='blue' else red_flag
                        enemy_flag.carried_by=None
                        enemy_flag.pos=e.pos
                        e.carrying_flag=False
                    e.respawn_ticks = RESPAWN_TICKS
                    # shaped reward for home defense
                    ag.brain.add_reward(0.05)
                    if DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                        print(f"[tick {tick}] {ag.team} KILL on {e.team} at {e.pos}")
                else:
                    # stun
                    if e.carrying_flag:
                        enemy_flag = blue_flag if e.team=='blue' else red_flag
                        enemy_flag.carried_by=None
                        enemy_flag.pos=e.pos
                        e.carrying_flag=False
                    if e.stun_ticks<=0:
                        e.stun_ticks=STUN_TICKS
                        # small reward for stunning
                        ag.brain.add_reward(0.03)
                        if DEBUG_PER_TICK and tick<=DEBUG_TICKS:
                            print(f"[tick {tick}] {ag.team} STUN on {e.team} at {e.pos}")

    # End of episode handling if scored
    if scored:
        # extra reward to the scorer
        scorer.brain.add_reward(1.0)
        # team rewards and end episode
        win_blue = scored_team == 'blue'
        for ag in blue_agents:
            ag.brain.add_reward(0.5 if win_blue else -0.5)
            ag.brain.end_episode(win_blue)
            ag.tactic = ag.brain.choose()
        for ag in red_agents:
            ag.brain.add_reward(-0.5 if win_blue else 0.5)
            ag.brain.end_episode(not win_blue)
            ag.tactic = ag.brain.choose()

    # tick down yield cooldowns
    for ag in all_agents:
        if getattr(ag,'yield_cooldown',0) > 0:
            ag.yield_cooldown -= 1

    # Render
    draw(win, grid, blue_agents, red_agents, blue_flag, red_flag)

pygame.quit()