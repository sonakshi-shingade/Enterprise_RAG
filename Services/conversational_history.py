import json
import threading
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ConversationHistory:
	"""
	Thread-safe in-memory conversational history with optional JSON persistence.

	Message format:
	{
		"id": str,
		"role": "user" | "assistant" | "system",
		"content": str,
		"timestamp": ISO8601 str,
		"metadata": Optional[dict]
	}
	"""

	def __init__(
		self,
		persist_path: Optional[str] = None,
		sqlite_path: Optional[str] = None,
		max_messages: int = 1000,
	):
		self._lock = threading.Lock()
		self._messages: List[Dict] = []
		self.max_messages = max_messages
		self.persist_path = Path(persist_path) if persist_path else None

		# SQLite persistence (optional)
		self.sqlite_path = Path(sqlite_path) if sqlite_path else None
		self._conn: Optional[sqlite3.Connection] = None

		if self.persist_path and self.persist_path.exists():
			try:
				self.load_from_file(self.persist_path)
			except Exception:
				# if loading fails, start with empty history
				self._messages = []

		if self.sqlite_path:
			try:
				self._init_db()
				# Load existing messages from DB into memory (respecting max_messages)
				self.load_from_db()
			except Exception:
				# ignore DB init failures but keep in-memory history
				self._conn = None

	# ---------------- Low-level operations ----------------
	def _now(self) -> str:
		return datetime.utcnow().isoformat() + "Z"

	def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
		if role not in {"user", "assistant", "system"}:
			raise ValueError("role must be one of 'user', 'assistant', or 'system'")

		message = {
			"id": str(uuid.uuid4()),
			"role": role,
			"content": content,
			"timestamp": self._now(),
			"metadata": metadata or {},
		}

		with self._lock:
			self._messages.append(message)
			# enforce max size
			if len(self._messages) > self.max_messages:
				# drop oldest
				excess = len(self._messages) - self.max_messages
				self._messages = self._messages[excess:]

			# Persist to SQLite if enabled
			if self._conn:
				try:
					self._db_insert_message(message)
				except Exception:
					# don't fail on DB errors; history remains in-memory
					pass

		return message

	def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> Dict:
		return self.add_message("user", content, metadata)

	def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> Dict:
		return self.add_message("assistant", content, metadata)

	def add_system_message(self, content: str, metadata: Optional[Dict] = None) -> Dict:
		return self.add_message("system", content, metadata)

	# ---------------- Queries ----------------
	def get_messages(self, role: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
		with self._lock:
			msgs = list(self._messages)

		if role:
			msgs = [m for m in msgs if m.get("role") == role]

		if limit is not None:
			return msgs[-limit:]

		return msgs

	def get_formatted_for_llm(self, include_system: bool = True, limit: Optional[int] = None) -> List[Dict]:
		"""Return messages in the format expected by LLM services: list of {role, content}."""
		msgs = self.get_messages(limit=limit)
		if not include_system:
			msgs = [m for m in msgs if m.get("role") != "system"]

		return [{"role": m["role"], "content": m["content"]} for m in msgs]

	def clear(self) -> None:
		with self._lock:
			self._messages = []

	# ---------------- Persistence ----------------
	def save_to_file(self, path: Optional[str] = None) -> None:
		target = Path(path) if path else self.persist_path
		if not target:
			raise ValueError("No persist path configured; provide `path` or set `persist_path`.")

		target.parent.mkdir(parents=True, exist_ok=True)

		with self._lock:
			data = {"messages": self._messages}

		with target.open("w", encoding="utf-8") as fh:
			json.dump(data, fh, ensure_ascii=False, indent=2)

	# ---------------- SQLite helpers ----------------
	def _init_db(self) -> None:
		# ensure parent dir exists
		self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
		# connect (allow access from multiple threads)
		self._conn = sqlite3.connect(
			str(self.sqlite_path), check_same_thread=False
		)
		self._conn.row_factory = sqlite3.Row
		cur = self._conn.cursor()
		cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS conversations (
				id TEXT PRIMARY KEY,
				role TEXT NOT NULL,
				content TEXT NOT NULL,
				timestamp TEXT NOT NULL,
				metadata TEXT
			)
			"""
		)
		cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)")
		self._conn.commit()

	def _db_insert_message(self, message: Dict) -> None:
		if not self._conn:
			return

		cur = self._conn.cursor()
		cur.execute(
			"INSERT OR REPLACE INTO conversations (id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
			(
				message.get("id"),
				message.get("role"),
				message.get("content"),
				message.get("timestamp"),
				json.dumps(message.get("metadata", {}), ensure_ascii=False),
			),
		)
		self._conn.commit()

	def load_from_db(self, limit: Optional[int] = None) -> None:
		if not self._conn:
			return

		cur = self._conn.cursor()
		q = "SELECT id, role, content, timestamp, metadata FROM conversations ORDER BY timestamp ASC"
		if limit:
			q = q + " LIMIT ?"
			rows = cur.execute(q, (limit,)).fetchall()
		else:
			rows = cur.execute(q).fetchall()

		msgs: List[Dict] = []
		for r in rows:
			meta = {}
			try:
				meta = json.loads(r[4]) if r[4] else {}
			except Exception:
				meta = {}

			msgs.append(
				{
					"id": r[0],
					"role": r[1],
					"content": r[2],
					"timestamp": r[3],
					"metadata": meta,
				}
			)

		with self._lock:
			# keep only the most recent `max_messages` if necessary
			if len(msgs) > self.max_messages:
				msgs = msgs[-self.max_messages:]
			self._messages = msgs

	def close(self) -> None:
		if self._conn:
			try:
				self._conn.close()
			except Exception:
				pass
			finally:
				self._conn = None

	def load_from_file(self, path: Optional[str] = None) -> None:
		target = Path(path) if path else self.persist_path
		if not target or not target.exists():
			return

		with target.open("r", encoding="utf-8") as fh:
			data = json.load(fh)

		msgs = data.get("messages", [])
		with self._lock:
			# basic validation
			self._messages = [m for m in msgs if "role" in m and "content" in m]
			# enforce max_messages
			if len(self._messages) > self.max_messages:
				self._messages = self._messages[-self.max_messages:]

	def to_dict(self) -> Dict:
		with self._lock:
			return {"messages": list(self._messages)}

	def to_json(self) -> str:
		return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# Module-level singleton for easy reuse across the app
DEFAULT_PERSIST_PATH = Path("Database/conversations.json")
DEFAULT_SQLITE_PATH = Path("Database/conversations.sqlite3")
conversation_history = ConversationHistory(
	persist_path=str(DEFAULT_PERSIST_PATH),
	sqlite_path=str(DEFAULT_SQLITE_PATH),
	max_messages=2000,
)


if __name__ == "__main__":
	# Quick demo
	ch = ConversationHistory()
	ch.add_system_message("You are a helpful assistant.")
	ch.add_user_message("What is RAG?")
	ch.add_assistant_message("RAG stands for Retrieval-Augmented Generation.")
	print(ch.to_json())
