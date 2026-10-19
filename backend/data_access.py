"""
FinGraph Data Access Layer.

Provides fast, indexed, parameterized querying over the SQLite transaction store.
Guarantees safe pagination, input validation, and graph subgraph generation
without reading the 494 MB CSV into memory.
"""

import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "transactions.db"

SORTABLE_COLUMNS = {
    "id": "id",
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "nameOrig",
    "nameDest": "nameDest",
    "oldbalanceOrg": "oldbalanceOrg",
    "newbalanceOrig": "newbalanceOrig",
    "oldbalanceDest": "oldbalanceDest",
    "newbalanceDest": "newbalanceDest",
    "isFraud": "isFraud",
}


class DataAccess:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def set_db_path(self, path: Path):
        self.db_path = Path(path)

    def is_ready(self) -> bool:
        if not self.db_path.exists():
            return False
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                cur = conn.cursor()
                cur.execute("SELECT status FROM _import_metadata WHERE id = 1;")
                row = cur.fetchone()
                return bool(row and row[0] == "COMPLETE")
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        if not self.is_ready():
            return {"ready": False, "row_count": 0, "fraud_count": 0}
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT row_count, completed_at FROM _import_metadata WHERE id = 1;")
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM transactions WHERE isFraud = 1;")
            fraud_row = cur.fetchone()
            fraud_count = fraud_row[0] if fraud_row else 0
            return {
                "ready": True,
                "row_count": row[0] if row else 0,
                "fraud_count": fraud_count,
                "completed_at": row[1] if row else None,
                "db_size_mb": round(self.db_path.stat().st_size / (1024 * 1024), 2),
            }

    def _get_connection(self) -> sqlite3.Connection:
        if not self.is_ready():
            raise FileNotFoundError(
                f"Database not initialized or incomplete. Please run 'python src/init_db.py'."
            )
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def search_transactions(
        self,
        transaction_id: Optional[int] = None,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        account: Optional[str] = None,
        type: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        is_fraud: Optional[int] = None,
        step: Optional[int] = None,
        sort_by: str = "id",
        sort_dir: str = "asc",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        sort_col = SORTABLE_COLUMNS.get(sort_by, "id")
        sort_order = "DESC" if str(sort_dir).upper() == "DESC" else "ASC"

        conditions: List[str] = []
        params: List[Any] = []

        if transaction_id is not None:
            conditions.append("id = ?")
            params.append(int(transaction_id))

        if sender:
            conditions.append("nameOrig = ?")
            params.append(sender.strip())

        if receiver:
            conditions.append("nameDest = ?")
            params.append(receiver.strip())

        if account:
            conditions.append("(nameOrig = ? OR nameDest = ?)")
            params.extend([account.strip(), account.strip()])

        if type:
            conditions.append("type = ?")
            params.append(type.strip().upper())

        if min_amount is not None:
            conditions.append("amount >= ?")
            params.append(float(min_amount))

        if max_amount is not None:
            conditions.append("amount <= ?")
            params.append(float(max_amount))

        if is_fraud is not None:
            conditions.append("isFraud = ?")
            params.append(1 if int(is_fraud) == 1 else 0)

        if step is not None:
            conditions.append("step = ?")
            params.append(int(step))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cur = conn.cursor()

            # Count query
            count_sql = f"SELECT COUNT(*) FROM transactions {where_clause};"
            cur.execute(count_sql, params)
            total = cur.fetchone()[0]

            total_pages = math.ceil(total / page_size) if total > 0 else 1
            offset = (page - 1) * page_size

            # Items query
            items_sql = f"""
            SELECT id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                   nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
            FROM transactions
            {where_clause}
            ORDER BY {sort_col} {sort_order}
            LIMIT ? OFFSET ?;
            """
            cur.execute(items_sql, params + [page_size, offset])
            rows = cur.fetchall()

            items = [dict(row) for row in rows]

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "items": items,
            }

    def get_transaction_by_id(self, tx_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                       nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
                FROM transactions WHERE id = ?;
                """,
                (tx_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_account_graph(
        self, account_id: str, hops: int = 1, limit: int = 50
    ) -> Dict[str, Any]:
        account_id = account_id.strip()
        limit = max(1, min(100, limit))
        hops = 2 if hops == 2 else 1

        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1-Hop query: direct incoming and outgoing transactions
            cur.execute(
                """
                SELECT id, step, type, amount, nameOrig, nameDest, isFraud
                FROM transactions
                WHERE nameOrig = ? OR nameDest = ?
                ORDER BY isFraud DESC, amount DESC
                LIMIT ?;
                """,
                (account_id, account_id, limit),
            )
            hop1_rows = cur.fetchall()

            tx_records = [dict(r) for r in hop1_rows]
            seen_tx_ids = {r["id"] for r in tx_records}

            # If 2 hops requested, expand to neighbors
            if hops == 2 and tx_records:
                neighbors = set()
                for r in tx_records:
                    if r["nameOrig"] != account_id:
                        neighbors.add(r["nameOrig"])
                    if r["nameDest"] != account_id:
                        neighbors.add(r["nameDest"])

                # Query transactions between or from these neighbors
                if neighbors:
                    neighbor_list = list(neighbors)[:20]  # cap to top 20 neighbors
                    placeholders = ",".join(["?"] * len(neighbor_list))
                    remaining_limit = max(10, limit - len(tx_records))
                    cur.execute(
                        f"""
                        SELECT id, step, type, amount, nameOrig, nameDest, isFraud
                        FROM transactions
                        WHERE (nameOrig IN ({placeholders}) OR nameDest IN ({placeholders}))
                          AND id NOT IN ({','.join('?' for _ in seen_tx_ids)})
                        ORDER BY isFraud DESC, amount DESC
                        LIMIT ?;
                        """,
                        neighbor_list + neighbor_list + list(seen_tx_ids) + [remaining_limit],
                    )
                    for r in cur.fetchall():
                        tx_records.append(dict(r))

            return self._build_graph_payload(tx_records, root_account=account_id)

    def get_connected_network(self, limit: int = 40, include_fraud: bool = True) -> Dict[str, Any]:
        limit = max(15, min(80, limit))
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Find a cluster of connected transactions around high-impact fraud transactions
            if include_fraud:
                cur.execute(
                    """
                    SELECT id, step, type, amount, nameOrig, nameDest, isFraud
                    FROM transactions
                    WHERE isFraud = 1
                    ORDER BY amount DESC
                    LIMIT 15;
                    """
                )
                seed_fraud = [dict(r) for r in cur.fetchall()]
                seed_accounts = set()
                for r in seed_fraud:
                    seed_accounts.add(r["nameOrig"])
                    seed_accounts.add(r["nameDest"])

                tx_records = list(seed_fraud)
                seen_tx = {r["id"] for r in tx_records}

                if seed_accounts:
                    accounts_sample = list(seed_accounts)[:15]
                    placeholders = ",".join(["?"] * len(accounts_sample))
                    cur.execute(
                        f"""
                        SELECT id, step, type, amount, nameOrig, nameDest, isFraud
                        FROM transactions
                        WHERE (nameOrig IN ({placeholders}) OR nameDest IN ({placeholders}))
                        ORDER BY step ASC, amount DESC
                        LIMIT ?;
                        """,
                        accounts_sample + accounts_sample + [limit],
                    )
                    for r in cur.fetchall():
                        if r["id"] not in seen_tx:
                            tx_records.append(dict(r))
                            seen_tx.add(r["id"])
            else:
                cur.execute(
                    """
                    SELECT id, step, type, amount, nameOrig, nameDest, isFraud
                    FROM transactions
                    ORDER BY amount DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
                tx_records = [dict(r) for r in cur.fetchall()]

            return self._build_graph_payload(tx_records, root_account=None)

    def _build_graph_payload(
        self, tx_records: List[Dict[str, Any]], root_account: Optional[str] = None
    ) -> Dict[str, Any]:
        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        total_volume = 0.0
        fraud_edge_count = 0

        for r in tx_records:
            orig = r["nameOrig"]
            dest = r["nameDest"]
            amt = float(r["amount"])
            is_fraud = bool(r["isFraud"])

            total_volume += amt
            if is_fraud:
                fraud_edge_count += 1

            for acc, role in [(orig, "out"), (dest, "in")]:
                if acc not in nodes_map:
                    nodes_map[acc] = {
                        "id": acc,
                        "label": acc,
                        "isFraud": False,
                        "isRoot": (acc == root_account),
                        "inDegree": 0,
                        "outDegree": 0,
                        "totalInAmount": 0.0,
                        "totalOutAmount": 0.0,
                    }
                if role == "in":
                    nodes_map[acc]["inDegree"] += 1
                    nodes_map[acc]["totalInAmount"] += amt
                else:
                    nodes_map[acc]["outDegree"] += 1
                    nodes_map[acc]["totalOutAmount"] += amt

                if is_fraud:
                    nodes_map[acc]["isFraud"] = True

            edges.append({
                "id": f"tx-{r['id']}",
                "source": orig,
                "target": dest,
                "amount": amt,
                "type": r["type"],
                "step": int(r["step"]),
                "isFraud": is_fraud,
            })

        nodes_list = list(nodes_map.values())
        fraud_nodes_count = sum(1 for n in nodes_list if n["isFraud"])

        stats = {
            "totalNodes": len(nodes_list),
            "totalEdges": len(edges),
            "fraudNodes": fraud_nodes_count,
            "fraudEdges": fraud_edge_count,
            "totalVolume": round(total_volume, 2),
        }

        return {
            "root": root_account,
            "nodes": nodes_list,
            "edges": edges,
            "stats": stats,
            "isSample": False,
        }


data_access = DataAccess()
