#!/usr/bin/env python3

import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from operator import attrgetter
from typing import Any, Self

TICK_MS = 230  
MAX_DT = 16  
IDLE_MS = 20_000  # cap on when to end a fight


class Kind(StrEnum):
    SELF = "Self"
    PLAYER = "Player"
    SPAWN = "Spawn"
    CAST = "Cast"
    HIT = "Hit"
    CAST_END = "CastEnd"
    DEATH = "Death"


class Role(StrEnum):
    SELF = "SELF"
    MOB = "MOB"
    PC = "PC"


class Act(StrEnum):
    CAST = "CAST"
    HIT = "HIT"
    CRIT = "CRIT"
    END = "END"
    DEATH = "DEATH"


class End(StrEnum):
    MOB_DEATH = "mob death"
    YOUR_DEATH = "your death"
    IDLE = "20 s idle"
    EOF = "end of recording"


class Skip(StrEnum):
    UNDECODED = "undecoded"
    NOT_YOURS = "not yours"
    NO_FIGHT = "buff with no fight"
    LATE = "after its fight ended"


@dataclass(frozen=True, slots=True)
class Cast:
    t: int
    actor: int
    target: int
    skill: int

    @classmethod
    def from_json(cls, t: int, e: dict[str, Any]) -> Self:
        return cls(t, e["Actor"], e["Target"], e["Skill"])


@dataclass(frozen=True, slots=True)
class Hit:
    t: int
    actor: int
    target: int
    skill: int
    damage: int
    crit: bool

    @classmethod
    def from_json(cls, t: int, e: dict[str, Any]) -> Self:
        return cls(t, e["Actor"], e["Target"], e["Skill"], e["Damage"], e.get("Type") == 3)


@dataclass(frozen=True, slots=True)
class CastEnd:
    t: int
    actor: int
    skill: int

    @classmethod
    def from_json(cls, t: int, e: dict[str, Any]) -> Self:
        return cls(t, e["Actor"], e["Skill"])


@dataclass(frozen=True, slots=True)
class Death:
    t: int
    entity: int

    @classmethod
    def from_json(cls, t: int, e: dict[str, Any]) -> Self:
        return cls(t, e["Entity"])


type Event = Cast | Hit | CastEnd | Death
EVENTS: dict[Kind, type[Event]] = {Kind.CAST: Cast, Kind.HIT: Hit, Kind.CAST_END: CastEnd, Kind.DEATH: Death}


@dataclass(frozen=True, slots=True)
class Recording:
    me: int
    npc: dict[int, int]
    players: frozenset[int]
    events: tuple[Event, ...]
    undecoded: int

    @classmethod
    def read(cls, lines: Iterable[str]) -> Self:
        me, npc, players, events, undecoded = None, {}, set(), [], 0
        for line in lines:
            if not line.strip():
                continue
            m = json.loads(line)
            if (name := m.get("name")) not in Kind:
                continue
            kind = Kind(name)
            if (e := m.get("event")) is None:
                undecoded += 1
            elif kind in (Kind.SELF, Kind.PLAYER):
                players.add(e["Entity"])
                if e.get("Self"):
                    me = e["Entity"]
            elif kind == Kind.SPAWN:
                npc[e["Entity"]] = e["NPC"]
            else:
                events.append(EVENTS[kind].from_json(m["t"], e))

        if me is None:  # a recording started mid-session: you are whoever cast the most (@TODO: figure out a better way in a2kit itself maybe?)
            casts = Counter(ev.actor for ev in events if isinstance(ev, Cast))
            if not casts:
                sys.exit("prep: no casts, so no fights")
            me = casts.most_common(1)[0][0]

        events.sort(key=attrgetter("t"))
        return cls(me, npc, frozenset(players), tuple(events), undecoded)

    def is_mob(self, x: int) -> bool:
        return bool(x) and x != self.me and x not in self.players


@dataclass(slots=True)
class Fight:
    words: list[str]
    last: int

    def add(self, t: int, *words: str) -> None:
        dt = min(MAX_DT, max(0, math.floor((t - self.last) / TICK_MS + 0.5)))
        self.words += [f"DT_{dt}", *words]
        self.last = t

def dmg(d: int) -> str:
    return f"DMG_{math.ceil(2 ** (math.floor(4 * math.log2(d)) / 4))}" if d > 0 else "DMG_0"


@dataclass(slots=True)
class Prep:
    rec: Recording
    fights: dict[int, Fight] = field(default_factory=dict)  # the open ones, keyed by mob
    target: int | None = None  # the mob you last cast at or hit
    cast_in: dict[tuple[int, int], int] = field(default_factory=dict)  # keyed by (actor, skill) -> its Cast's mob
    lines: list[str] = field(default_factory=list)  # the closed fights
    ended: Counter[End] = field(default_factory=Counter)
    skipped: Counter[Skip] = field(default_factory=Counter)

    def run(self) -> Self:
        self.skipped[Skip.UNDECODED] = self.rec.undecoded
        for ev in self.rec.events:
            self.step(ev)
        for mob in list(self.fights):
            self.close(mob, End.EOF)
        return self

    def step(self, ev: Event) -> None:
        for mob in [m for m, f in self.fights.items() if ev.t - f.last > IDLE_MS]:
            self.close(mob, End.IDLE)
        
        # @REVIEW: are matches in Python really worth this quality slop?
        match ev:
            case Death(entity=x) if x == self.rec.me:
                for mob in list(self.fights):
                    self.fights[mob].add(ev.t, Role.SELF, Act.DEATH)
                    self.close(mob, End.YOUR_DEATH)
            case Death(entity=x) if x in self.fights:
                self.fights[x].add(ev.t, Role.MOB, Act.DEATH)
                self.close(x, End.MOB_DEATH)
            case Death():
                self.skipped[Skip.NOT_YOURS] += 1
            case CastEnd(actor=a, skill=s):
                mob = self.cast_in.pop((a, s), None)
                if mob is None:
                    self.skipped[Skip.NOT_YOURS] += 1
                elif mob not in self.fights:
                    self.skipped[Skip.LATE] += 1
                else:
                    self.fights[mob].add(ev.t, self.role(a, mob), Act.END, f"SK_{s}")
            case Cast(actor=a, target=b, skill=s):
                if (mob := self.route(ev.t, a, b)) is not None:
                    self.fights[mob].add(ev.t, self.role(a, mob), Act.CAST, f"SK_{s}", self.role(b, mob))
                    self.cast_in[(a, s)] = mob
            case Hit(actor=a, target=b, skill=s, damage=d, crit=c):
                if (mob := self.route(ev.t, a, b)) is not None:
                    act = Act.CRIT if c else Act.HIT
                    self.fights[mob].add(ev.t, self.role(a, mob), act, f"SK_{s}", dmg(d), self.role(b, mob))

    def route(self, t: int, a: int, b: int) -> int | None:
        rec = self.rec
        if a == rec.me and rec.is_mob(b):
            self.target = b
            return self.start(b, t)
        if b == rec.me and rec.is_mob(a):
            return self.start(a, t)
        if a == rec.me:  # on yourself or another player, e.g. a buff or a heal
            if self.target in self.fights:
                return self.target
            self.skipped[Skip.NO_FIGHT] += 1
            return None
        if rec.is_mob(b) and b in self.fights:
            return b
        if rec.is_mob(a) and a in self.fights:
            return a
        self.skipped[Skip.NOT_YOURS] += 1
        return None

    def start(self, mob: int, t: int) -> int:
        if mob not in self.fights:
            self.fights[mob] = Fight(["BOS", f"NPC_{self.rec.npc.get(mob, '?')}"], t)
        return mob

    def role(self, x: int, mob: int) -> Role:
        return Role.SELF if x == self.rec.me else Role.MOB if x == mob else Role.PC

    def close(self, mob: int, how: End) -> None:
        self.lines.append(" ".join([*self.fights.pop(mob).words, "EOS"]))
        self.ended[how] += 1


def tally[K: StrEnum](c: Counter[K]) -> str:
    return ", ".join(f"{n} {k}" for k, n in c.most_common() if n)


def main() -> None:
    prep = Prep(Recording.read(sys.stdin)).run()
    for line in prep.lines:
        print(line)
    print(f"prep: {len(prep.lines)} fights ({tally(prep.ended)}); skipped {tally(prep.skipped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
