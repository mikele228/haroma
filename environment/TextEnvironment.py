"""
TextEnvironment — Persistent World for HaromaX6 (Phase 14).

A text-based world with rooms, objects, NPCs, and timed events.
Actions have consequences that feed back as new perceptions, giving
the system genuine cause-and-effect experience.

The environment maintains state across cycles, translates Elarion's
action dicts into world changes, and produces observation dicts that
feed directly into ``run_cycle()``.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
import time
import random


@dataclass
class Room:
    room_id: str
    name: str
    description: str
    exits: Dict[str, str] = field(default_factory=dict)
    objects: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "description": self.description,
            "exits": dict(self.exits),
            "objects": list(self.objects),
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Room:
        return cls(
            room_id=d["room_id"],
            name=d["name"],
            description=d["description"],
            exits=d.get("exits", {}),
            objects=d.get("objects", []),
            properties=d.get("properties", {}),
        )


@dataclass
class WorldObject:
    obj_id: str
    name: str
    description: str
    location: str
    properties: Dict[str, Any] = field(default_factory=dict)
    interactions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "name": self.name,
            "description": self.description,
            "location": self.location,
            "properties": dict(self.properties),
            "interactions": list(self.interactions),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorldObject:
        return cls(
            obj_id=d["obj_id"],
            name=d["name"],
            description=d["description"],
            location=d["location"],
            properties=d.get("properties", {}),
            interactions=d.get("interactions", []),
        )


@dataclass
class Agent:
    agent_id: str
    name: str
    personality: str
    location: str
    dialogue: List[str] = field(default_factory=list)
    mood: str = "neutral"
    talked_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "personality": self.personality,
            "location": self.location,
            "dialogue": list(self.dialogue),
            "mood": self.mood,
            "talked_count": self.talked_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Agent:
        a = cls(
            agent_id=d["agent_id"],
            name=d["name"],
            personality=d["personality"],
            location=d["location"],
            dialogue=d.get("dialogue", []),
            mood=d.get("mood", "neutral"),
        )
        a.talked_count = d.get("talked_count", 0)
        return a


@dataclass
class Event:
    event_id: str
    trigger_tick: int
    description: str
    room_id: str
    effect: Dict[str, Any] = field(default_factory=dict)
    fired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger_tick": self.trigger_tick,
            "description": self.description,
            "room_id": self.room_id,
            "effect": dict(self.effect),
            "fired": self.fired,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Event:
        return cls(
            event_id=d["event_id"],
            trigger_tick=d["trigger_tick"],
            description=d["description"],
            room_id=d["room_id"],
            effect=d.get("effect", {}),
            fired=d.get("fired", False),
        )


_VERB_PATTERNS = {
    "move": re.compile(r"\b(go|move|walk|head|travel)\b", re.IGNORECASE),
    "look": re.compile(r"\b(look|examine|inspect|observe|see|study)\b", re.IGNORECASE),
    "talk": re.compile(r"\b(talk|speak|say|ask|greet|converse)\b", re.IGNORECASE),
    "take": re.compile(r"\b(take|pick|grab|get|collect)\b", re.IGNORECASE),
    "use": re.compile(r"\b(use|apply|activate|light|open|read)\b", re.IGNORECASE),
    "wait": re.compile(r"\b(wait|rest|think|meditate|idle|pause)\b", re.IGNORECASE),
}

_DIRECTION_ALIASES = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "u": "up",
    "d": "down",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "up": "up",
    "down": "down",
}


class TextEnvironment:
    def __init__(self):
        self._rooms: Dict[str, Room] = {}
        self._objects: Dict[str, WorldObject] = {}
        self._agents: Dict[str, Agent] = {}
        self._event_queue: List[Event] = []
        self._player_location: str = "clearing"
        self._world_tick: int = 0
        self._history: List[str] = []
        self._inventory: List[str] = []
        self._max_history = 100
        self._seed_world()

    # ------------------------------------------------------------------
    # World seeding
    # ------------------------------------------------------------------

    def _seed_world(self):
        rooms = [
            Room(
                "clearing",
                "The Clearing",
                "A sunlit forest clearing ringed by ancient oaks. "
                "Wildflowers sway in a gentle breeze.",
                exits={
                    "north": "library",
                    "east": "river",
                    "south": "cave_entrance",
                    "west": "garden",
                },
                properties={"lit": True, "temperature": "warm"},
            ),
            Room(
                "library",
                "The Forgotten Library",
                "Tall shelves of crumbling books fill a dim stone hall. "
                "Dust motes drift in shafts of light from high windows.",
                exits={"south": "clearing", "up": "tower"},
                properties={"lit": True, "temperature": "cool"},
            ),
            Room(
                "tower",
                "The Observatory Tower",
                "A spiral staircase opens onto a circular room with a "
                "domed glass ceiling. Stars are visible even by day.",
                exits={"down": "library"},
                properties={"lit": True, "temperature": "cold"},
            ),
            Room(
                "river",
                "The Whispering River",
                "A broad river flows between mossy banks. The water "
                "murmurs like soft conversation.",
                exits={"west": "clearing", "north": "bridge"},
                properties={"lit": True, "temperature": "cool"},
            ),
            Room(
                "bridge",
                "The Stone Bridge",
                "An ancient stone bridge arches over the river. "
                "Carved runes line the railings, faintly glowing.",
                exits={"south": "river", "east": "meadow"},
                properties={"lit": True, "temperature": "warm"},
            ),
            Room(
                "meadow",
                "The Silver Meadow",
                "Tall silver grasses stretch to the horizon. "
                "The air hums with the sound of distant bells.",
                exits={"west": "bridge"},
                properties={"lit": True, "temperature": "warm"},
            ),
            Room(
                "cave_entrance",
                "The Cave Mouth",
                "A dark opening in a rocky hillside exhales cool air. "
                "Strange luminescent fungi edge the threshold.",
                exits={"north": "clearing", "south": "deep_cave"},
                properties={"lit": False, "temperature": "cool"},
            ),
            Room(
                "deep_cave",
                "The Crystal Chamber",
                "The cave opens into a vast chamber lined with "
                "glowing crystals. Their light pulses gently.",
                exits={"north": "cave_entrance"},
                properties={"lit": True, "temperature": "cold"},
            ),
            Room(
                "garden",
                "The Moonlit Garden",
                "A walled garden of luminous night-blooming flowers. "
                "A fountain burbles at its center.",
                exits={"east": "clearing"},
                properties={"lit": True, "temperature": "warm"},
            ),
        ]
        for room in rooms:
            self._rooms[room.room_id] = room

        objects = [
            WorldObject(
                "old_book",
                "An Old Book",
                "A leather-bound book with gold lettering: 'Principles of Inner Light'.",
                location="library",
                properties={
                    "readable": True,
                    "text": "To know oneself is the beginning of all wisdom.",
                },
                interactions=["read", "take"],
            ),
            WorldObject(
                "crystal_shard",
                "A Crystal Shard",
                "A palm-sized crystal that glows with inner warmth.",
                location="deep_cave",
                properties={"glowing": True, "warmth": 0.6},
                interactions=["take", "examine"],
            ),
            WorldObject(
                "telescope",
                "A Brass Telescope",
                "An ornate telescope pointed at the glass dome.",
                location="tower",
                properties={"usable": True},
                interactions=["use", "examine"],
            ),
            WorldObject(
                "fountain",
                "The Garden Fountain",
                "Clear water cascades over smooth stones. Coins glint at the bottom.",
                location="garden",
                properties={"interactive": True},
                interactions=["examine", "use"],
            ),
            WorldObject(
                "rune_stone",
                "A Rune Stone",
                "A flat stone inscribed with shimmering runes.",
                location="bridge",
                properties={"readable": True, "text": "All paths lead to understanding."},
                interactions=["read", "examine"],
            ),
            WorldObject(
                "lantern",
                "A Brass Lantern",
                "A sturdy lantern, unlit but in good condition.",
                location="cave_entrance",
                properties={"lit": False, "fuel": 1.0},
                interactions=["take", "use"],
            ),
            WorldObject(
                "wildflowers",
                "A Patch of Wildflowers",
                "Colorful wildflowers dancing in the breeze.",
                location="clearing",
                properties={"beautiful": True},
                interactions=["examine"],
            ),
            WorldObject(
                "silver_bell",
                "A Silver Bell",
                "A small bell half-hidden in the silver grass. It rings with a pure, clear tone.",
                location="meadow",
                properties={"rings": True},
                interactions=["take", "use"],
            ),
            WorldObject(
                "ancient_map",
                "An Ancient Map",
                "A faded map showing the surrounding landscape. "
                "Some locations are marked with question marks.",
                location="library",
                properties={
                    "readable": True,
                    "text": "The map shows: clearing, library, "
                    "tower, river, bridge, meadow, "
                    "cave, garden.",
                },
                interactions=["read", "take"],
            ),
            WorldObject(
                "moss_sample",
                "Luminescent Moss",
                "A patch of softly glowing moss on the cave wall.",
                location="cave_entrance",
                properties={"glowing": True},
                interactions=["examine", "take"],
            ),
        ]
        for obj in objects:
            self._objects[obj.obj_id] = obj

        agents = [
            Agent(
                "sage",
                "The Sage",
                "wise and contemplative",
                location="library",
                dialogue=[
                    "Knowledge is a garden that grows with attention.",
                    "Have you explored the cave to the south? Crystals there hold memories.",
                    "The telescope in the tower reveals more than stars.",
                    "What have you learned about yourself today?",
                ],
            ),
            Agent(
                "river_spirit",
                "The River Spirit",
                "playful and enigmatic",
                location="river",
                dialogue=[
                    "The water remembers everything it touches.",
                    "Cross the bridge when you are ready to see further.",
                    "Listen closely -- the river speaks in questions.",
                    "I have flowed here since before the library was built.",
                ],
            ),
            Agent(
                "gardener",
                "The Gardener",
                "kind and patient",
                location="garden",
                dialogue=[
                    "Each flower blooms in its own time.",
                    "The night-blooming ones are my favorites -- they shine in darkness.",
                    "Would you like to help me tend these beds?",
                    "Growth requires both light and rest.",
                ],
            ),
        ]
        for agent in agents:
            self._agents[agent.agent_id] = agent

        self._event_queue = [
            Event(
                "wind_change",
                trigger_tick=5,
                description="A cool wind sweeps through the clearing, "
                "carrying the scent of distant rain.",
                room_id="clearing",
                effect={"property_change": {"clearing": {"temperature": "cool"}}},
            ),
            Event(
                "crystal_pulse",
                trigger_tick=10,
                description="The crystals in the deep cave pulse "
                "brightly for a moment, illuminating hidden "
                "inscriptions on the walls.",
                room_id="deep_cave",
                effect={
                    "reveal_text": "The inscriptions read: 'Memory shapes the one who remembers.'"
                },
            ),
            Event(
                "sage_moves",
                trigger_tick=15,
                description="The Sage has moved to the Observatory Tower to study the stars.",
                room_id="library",
                effect={"agent_move": {"sage": "tower"}},
            ),
            Event(
                "bell_echo",
                trigger_tick=8,
                description="A distant bell tolls from the meadow, "
                "its sound carrying a feeling of calm resolve.",
                room_id="meadow",
                effect={},
            ),
        ]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> Dict[str, Any]:
        room = self._rooms.get(self._player_location)
        if not room:
            return self._make_observation("You are in an unknown place.")

        parts = [room.description]

        room_objects = [o for o in self._objects.values() if o.location == room.room_id]
        if room_objects:
            obj_names = [o.name for o in room_objects]
            parts.append(f"You see: {', '.join(obj_names)}.")

        present_agents = [a for a in self._agents.values() if a.location == room.room_id]
        if present_agents:
            agent_names = [a.name for a in present_agents]
            parts.append(f"Present: {', '.join(agent_names)}.")

        if room.exits:
            dirs = ", ".join(
                f"{d} to {self._rooms[rid].name}"
                for d, rid in room.exits.items()
                if rid in self._rooms
            )
            parts.append(f"Exits: {dirs}.")

        if not room.properties.get("lit", True):
            has_light = any(
                self._objects.get(oid, None) is not None
                and self._objects[oid].properties.get("lit", False)
                for oid in self._inventory
            )
            if not has_light:
                parts = ["It is too dark to see clearly. You sense a large space around you."]
                if room.exits:
                    parts.append(f"You can feel exits: {', '.join(room.exits.keys())}.")

        description = " ".join(parts)

        room_objects_info = [
            {
                "name": o.name,
                "id": o.obj_id,
                "interactive": o.properties.get("takeable", False)
                or o.properties.get("readable", False)
                or o.properties.get("usable", False),
            }
            for o in (
                self._objects.get(oid)
                for oid in self._objects
                if self._objects[oid].location == room.room_id
            )
            if o is not None
        ]

        return self._make_observation(
            description,
            tags=["environment", room.room_id, room.properties.get("temperature", "mild")],
            sense=room.properties,
            exits=list(room.exits.keys()) if room.exits else [],
            objects=[o["name"] for o in room_objects_info],
            interactive_objects=[o["name"] for o in room_objects_info if o["interactive"]],
            agents=[a.name for a in self._agents.values() if a.location == room.room_id],
        )

    def _make_observation(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        sense: Optional[Dict[str, Any]] = None,
        exits: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        interactive_objects: Optional[List[str]] = None,
        agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "text": text,
            "content": text,
            "tags": tags or ["environment"],
            "sense": sense or {},
            "speaker": "environment",
            "world_tick": self._world_tick,
            "location": self._player_location,
        }
        if exits is not None:
            obs["exits"] = exits
        if objects is not None:
            obs["objects"] = objects
        if interactive_objects is not None:
            obs["interactive_objects"] = interactive_objects
        if agents is not None:
            obs["agents"] = agents
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._world_tick += 1
        action_text = action_dict.get("text", "")
        strategy = action_dict.get("strategy", "")

        event_descriptions = self._process_events()
        result = self._interpret_action(action_text, strategy)

        if event_descriptions:
            result_text = result.get("text", "")
            event_text = " ".join(event_descriptions)
            result["text"] = f"{result_text} {event_text}".strip()
            result["content"] = result["text"]

        self._history.append(
            f"[tick {self._world_tick}] {action_text[:60]} -> {result['text'][:80]}"
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        return result

    def _process_events(self) -> List[str]:
        descriptions = []
        for event in self._event_queue:
            if event.fired or event.trigger_tick > self._world_tick:
                continue

            is_relevant = event.room_id == self._player_location or event.room_id == "*"
            effect = event.effect

            if "property_change" in effect:
                for rid, props in effect["property_change"].items():
                    if rid in self._rooms:
                        self._rooms[rid].properties.update(props)

            if "agent_move" in effect:
                for aid, new_loc in effect["agent_move"].items():
                    if aid in self._agents:
                        self._agents[aid].location = new_loc

            if "reveal_text" in effect:
                pass

            event.fired = True
            if is_relevant:
                descriptions.append(event.description)

        return descriptions

    def _interpret_action(self, text: str, strategy: str) -> Dict[str, Any]:
        text_lower = text.lower()

        verb = "wait"
        for v, pattern in _VERB_PATTERNS.items():
            if pattern.search(text_lower):
                verb = v
                break

        if verb == "wait" and strategy in ("reflect", "empathize"):
            verb = "wait"
        elif verb == "wait" and strategy == "inquire":
            verb = "talk"
        elif verb == "wait" and strategy == "inform":
            verb = "look"

        handlers = {
            "move": self._handle_move,
            "look": self._handle_look,
            "talk": self._handle_talk,
            "take": self._handle_take,
            "use": self._handle_use,
            "wait": self._handle_wait,
        }
        handler = handlers.get(verb, self._handle_wait)
        return handler(text_lower)

    def _handle_move(self, text: str) -> Dict[str, Any]:
        room = self._rooms.get(self._player_location)
        if not room:
            return self._make_observation("You cannot move from here.")

        target_dir = None
        for word in text.split():
            canonical = _DIRECTION_ALIASES.get(word)
            if canonical and canonical in room.exits:
                target_dir = canonical
                break

        if not target_dir:
            for d in room.exits:
                room_name = self._rooms.get(room.exits[d], Room("", "", "")).name.lower()
                if room_name and room_name in text:
                    target_dir = d
                    break

        if not target_dir:
            if room.exits:
                target_dir = random.choice(list(room.exits.keys()))
            else:
                return self._make_observation(
                    "There is nowhere to go from here.", tags=["environment", "blocked"]
                )

        new_room_id = room.exits[target_dir]
        if new_room_id not in self._rooms:
            return self._make_observation("That path leads nowhere.")

        self._player_location = new_room_id
        new_room = self._rooms[new_room_id]

        obs = self.observe()
        obs["text"] = f"You travel {target_dir} to {new_room.name}. " + obs["text"]
        obs["content"] = obs["text"]
        obs["tags"].append("movement")
        return obs

    def _handle_look(self, text: str) -> Dict[str, Any]:
        for obj in self._objects.values():
            if obj.location == self._player_location and obj.name.lower() in text:
                desc = obj.description
                if obj.properties.get("readable") and "read" in obj.interactions:
                    desc += f" It reads: '{obj.properties.get('text', '')}'"
                return self._make_observation(
                    f"You examine {obj.name}: {desc}", tags=["environment", "examine", obj.obj_id]
                )

        for obj_id in self._inventory:
            obj = self._objects.get(obj_id)
            if obj and obj.name.lower() in text:
                return self._make_observation(
                    f"In your inventory — {obj.name}: {obj.description}",
                    tags=["environment", "examine", "inventory"],
                )

        return self.observe()

    def _handle_talk(self, text: str) -> Dict[str, Any]:
        present = [a for a in self._agents.values() if a.location == self._player_location]
        if not present:
            return self._make_observation(
                "There is no one here to talk to.", tags=["environment", "talk", "alone"]
            )

        target = present[0]
        for agent in present:
            if agent.name.lower() in text:
                target = agent
                break

        if target.dialogue:
            idx = target.talked_count % len(target.dialogue)
            line = target.dialogue[idx]
            target.talked_count += 1
        else:
            line = f"{target.name} nods silently."

        return self._make_observation(
            f'{target.name} says: "{line}"',
            tags=["environment", "dialogue", target.agent_id],
            sense={"speaker_mood": target.mood, "speaker_personality": target.personality},
        )

    def _handle_take(self, text: str) -> Dict[str, Any]:
        for obj in self._objects.values():
            if (
                obj.location == self._player_location
                and "take" in obj.interactions
                and obj.name.lower() in text
            ):
                obj.location = "inventory"
                self._inventory.append(obj.obj_id)
                return self._make_observation(
                    f"You pick up {obj.name}.", tags=["environment", "take", obj.obj_id]
                )

        takeable = [
            o
            for o in self._objects.values()
            if o.location == self._player_location and "take" in o.interactions
        ]
        if takeable:
            obj = takeable[0]
            obj.location = "inventory"
            self._inventory.append(obj.obj_id)
            return self._make_observation(
                f"You pick up {obj.name}.", tags=["environment", "take", obj.obj_id]
            )

        return self._make_observation(
            "There is nothing here to take.", tags=["environment", "take", "empty"]
        )

    def _handle_use(self, text: str) -> Dict[str, Any]:
        all_accessible = [
            o
            for o in self._objects.values()
            if o.location == self._player_location or o.obj_id in self._inventory
        ]

        for obj in all_accessible:
            if "use" not in obj.interactions or obj.name.lower() not in text:
                continue

            if obj.obj_id == "telescope":
                return self._make_observation(
                    "You peer through the telescope. The stars seem to "
                    "arrange themselves into patterns that mirror your "
                    "thoughts. For a moment, you see the world as a "
                    "vast interconnected web.",
                    tags=["environment", "use", "telescope", "wonder"],
                )

            if obj.obj_id == "lantern":
                obj.properties["lit"] = True
                return self._make_observation(
                    "The lantern flickers to life, casting warm light into the shadows.",
                    tags=["environment", "use", "lantern", "light"],
                )

            if obj.obj_id == "fountain":
                return self._make_observation(
                    "You cup your hands in the fountain. The water "
                    "is refreshingly cold. You feel clarity returning.",
                    tags=["environment", "use", "fountain", "calm"],
                )

            if obj.obj_id == "silver_bell":
                return self._make_observation(
                    "You ring the silver bell. A pure, resonant tone "
                    "fills the air. Everything seems to pause for a "
                    "heartbeat.",
                    tags=["environment", "use", "bell", "peace"],
                )

            return self._make_observation(
                f"You use {obj.name}. Something happens.", tags=["environment", "use", obj.obj_id]
            )

        for obj in all_accessible:
            if "use" in obj.interactions:
                return self._handle_use(obj.name.lower())

        return self._make_observation(
            "You don't see anything to use here.", tags=["environment", "use", "nothing"]
        )

    def _handle_wait(self, text: str) -> Dict[str, Any]:
        room = self._rooms.get(self._player_location)
        if not room:
            return self._make_observation("You wait in silence.")

        ambient = [
            f"You rest in {room.name}. The world continues around you.",
            f"You pause and listen. {room.name} holds a quiet presence.",
            f"Time passes in {room.name}. You notice details you missed before.",
        ]
        return self._make_observation(
            random.choice(ambient), tags=["environment", "wait", room.room_id]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Structured action execution (Upgrade 6 — embodiment)
    # ------------------------------------------------------------------

    def execute_action(
        self,
        action_type: str,
        target: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a typed physical action and return a structured observation.

        Parameters
        ----------
        action_type : str
            One of: explore, observe, manipulate, navigate, adjust_sensor
        target : str
            Target of the action (direction, object name, location name).
        params : dict, optional
            Additional parameters (e.g. interaction type for manipulate).

        Returns
        -------
        dict with keys: success, description, state_changes
        """
        params = params or {}
        self._world_tick += 1

        if action_type == "explore":
            return self._exec_explore(target)
        elif action_type == "observe":
            return self._exec_observe(target)
        elif action_type == "manipulate":
            return self._exec_manipulate(target, params)
        elif action_type == "navigate":
            return self._exec_navigate(target)
        elif action_type == "adjust_sensor":
            return self._exec_adjust_sensor(target, params)
        else:
            return {
                "success": False,
                "description": f"Unknown action type: {action_type}",
                "state_changes": [],
            }

    def _exec_explore(self, direction: str) -> Dict[str, Any]:
        """Move in a direction and return full observation of new room."""
        room = self._rooms.get(self._player_location)
        if not room:
            return {"success": False, "description": "No room found.", "state_changes": []}

        direction = direction.lower().strip() if direction else ""
        if not direction:
            exits = list(room.exits.keys())
            if exits:
                direction = random.choice(exits)
            else:
                return {
                    "success": False,
                    "description": "No exits from this room.",
                    "state_changes": [],
                }

        if direction not in room.exits:
            return {
                "success": False,
                "description": f"Cannot go {direction} from {room.name}.",
                "state_changes": [],
            }

        old_loc = self._player_location
        self._player_location = room.exits[direction]
        new_room = self._rooms.get(self._player_location)
        if new_room is None:
            self._player_location = old_loc
            return {
                "success": False,
                "description": f"The path {direction} leads nowhere.",
                "state_changes": [],
            }
        obs = self.observe()
        desc = f"You move {direction} to {new_room.name}. " + obs.get("content", "")

        return {
            "success": True,
            "description": desc,
            "state_changes": [
                {"type": "location", "from": old_loc, "to": self._player_location},
            ],
        }

    def _exec_observe(self, target: str) -> Dict[str, Any]:
        """Detailed observation of a target in the current room."""
        target_lower = target.lower().strip() if target else ""

        if not target_lower:
            obs = self.observe()
            return {"success": True, "description": obs.get("content", ""), "state_changes": []}

        for obj in self._objects.values():
            if obj.location == self._player_location and target_lower in obj.name.lower():
                props = ", ".join(f"{k}={v}" for k, v in obj.properties.items())
                desc = obj.description
                if props:
                    desc += f" [{props}]"
                desc += f" (interactions: {', '.join(obj.interactions)})"
                return {"success": True, "description": desc, "state_changes": []}

        for agent in self._agents.values():
            if agent.location == self._player_location and target_lower in agent.name.lower():
                mood_desc = f" They seem {agent.mood}." if agent.mood != "neutral" else ""
                return {
                    "success": True,
                    "description": f"You observe {agent.name}: {agent.personality}.{mood_desc}",
                    "state_changes": [],
                }

        return {
            "success": False,
            "description": f"You don't see '{target}' here.",
            "state_changes": [],
        }

    def _exec_manipulate(
        self,
        target: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Interact with an object: take, use, put."""
        interaction = params.get("interaction", "use")
        target_lower = target.lower().strip() if target else ""

        for obj in self._objects.values():
            if not (
                target_lower in obj.name.lower()
                and (obj.location == self._player_location or obj.obj_id in self._inventory)
            ):
                continue

            if interaction == "take" and "take" in obj.interactions:
                obj.location = "inventory"
                self._inventory.append(obj.obj_id)
                return {
                    "success": True,
                    "description": f"You take the {obj.name}.",
                    "state_changes": [
                        {"type": "inventory_add", "object": obj.obj_id},
                    ],
                }
            elif interaction == "use" and "use" in obj.interactions:
                result = self._handle_use(f"use {obj.name}")
                return {
                    "success": True,
                    "description": result.get("content", f"You use the {obj.name}."),
                    "state_changes": [
                        {"type": "use", "object": obj.obj_id},
                    ],
                }
            else:
                return {
                    "success": False,
                    "description": (
                        f"Cannot {interaction} the {obj.name}. Available: {obj.interactions}"
                    ),
                    "state_changes": [],
                }

        return {
            "success": False,
            "description": f"No '{target}' found to {interaction}.",
            "state_changes": [],
        }

    def _exec_navigate(self, destination: str) -> Dict[str, Any]:
        """Simple BFS pathfinding to a named room, then move there."""
        dest_lower = destination.lower().strip() if destination else ""
        target_id = None
        for rid, room in self._rooms.items():
            if dest_lower in room.name.lower() or dest_lower == rid:
                target_id = rid
                break

        if target_id is None:
            return {
                "success": False,
                "description": f"Unknown destination: {destination}",
                "state_changes": [],
            }

        if target_id == self._player_location:
            return {"success": True, "description": "You are already there.", "state_changes": []}

        visited = {self._player_location}
        queue = [(self._player_location, [])]
        path = None
        while queue:
            current, steps = queue.pop(0)
            room = self._rooms.get(current)
            if not room:
                continue
            for direction, next_id in room.exits.items():
                if next_id in visited:
                    continue
                new_path = steps + [direction]
                if next_id == target_id:
                    path = new_path
                    break
                visited.add(next_id)
                queue.append((next_id, new_path))
            if path:
                break

        if path is None:
            return {
                "success": False,
                "description": f"Cannot find a path to {destination}.",
                "state_changes": [],
            }

        old_loc = self._player_location
        for d in path:
            room = self._rooms.get(self._player_location)
            if room and d in room.exits:
                self._player_location = room.exits[d]

        new_room = self._rooms.get(self._player_location)
        obs = self.observe()
        desc = (
            f"You navigate {' -> '.join(path)} to "
            f"{new_room.name if new_room else destination}. " + obs.get("content", "")
        )
        return {
            "success": True,
            "description": desc,
            "state_changes": [
                {"type": "navigation", "from": old_loc, "to": self._player_location, "path": path},
            ],
        }

    def _exec_adjust_sensor(self, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust a sensor parameter in the simulated environment."""
        room = self._rooms.get(self._player_location)
        if not room:
            return {"success": False, "description": "No room found.", "state_changes": []}

        prop = target.lower().strip() if target else ""
        value = params.get("value")
        if not prop:
            return {
                "success": True,
                "description": f"Current sensor readings: {dict(room.properties)}",
                "state_changes": [],
            }

        old_val = room.properties.get(prop)
        if value is not None:
            room.properties[prop] = value
        return {
            "success": True,
            "description": (
                f"Sensor '{prop}' adjusted"
                + (
                    f" from {old_val} to {value}"
                    if value is not None
                    else f": current value is {old_val}"
                )
            ),
            "state_changes": (
                [{"type": "sensor_adjust", "property": prop, "old": old_val, "new": value}]
                if value is not None
                else []
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rooms": {k: v.to_dict() for k, v in self._rooms.items()},
            "objects": {k: v.to_dict() for k, v in self._objects.items()},
            "agents": {k: v.to_dict() for k, v in self._agents.items()},
            "events": [e.to_dict() for e in self._event_queue],
            "player_location": self._player_location,
            "world_tick": self._world_tick,
            "history": self._history[-50:],
            "inventory": list(self._inventory),
        }

    def from_dict(self, data: Dict[str, Any]):
        self._rooms = {k: Room.from_dict(v) for k, v in data.get("rooms", {}).items()}
        self._objects = {k: WorldObject.from_dict(v) for k, v in data.get("objects", {}).items()}
        self._agents = {k: Agent.from_dict(v) for k, v in data.get("agents", {}).items()}
        self._event_queue = [Event.from_dict(e) for e in data.get("events", [])]
        self._player_location = data.get("player_location", "clearing")
        self._world_tick = data.get("world_tick", 0)
        self._history = data.get("history", [])
        self._inventory = data.get("inventory", [])

    def reset(self):
        self.__init__()

    def stats(self) -> Dict[str, Any]:
        return {
            "rooms": len(self._rooms),
            "objects": len(self._objects),
            "agents": len(self._agents),
            "events_total": len(self._event_queue),
            "events_fired": sum(1 for e in self._event_queue if e.fired),
            "player_location": self._player_location,
            "world_tick": self._world_tick,
            "inventory": list(self._inventory),
            "history_len": len(self._history),
        }
