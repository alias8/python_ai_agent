"""Customer orders, and the one query the agent is allowed to run on them.

Seed local sample data (after `alembic upgrade head`):

    python -m agent_service.shared.orders
"""

from sqlalchemy import String, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from .db import Base, new_session


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. "ORD-1042"
    store_id: Mapped[str] = mapped_column(String)
    customer_id: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String)
    carrier: Mapped[str | None] = mapped_column(String)
    eta: Mapped[str | None] = mapped_column(String)  # ISO date


def find_customer_order(session: Session, order_id: str, customer_id: str, store_id: str) -> OrderRow | None:
    """The order, only if it belongs to this customer at this store.

    Ownership is part of the query itself, not a check made afterwards on a
    row fetched by ID alone — so there's no code path that holds another
    customer's order, and nothing to forget to check. An order that exists
    but isn't theirs comes back as None, exactly like one that doesn't
    exist, so the caller can't tell the two apart either.
    """
    return session.scalars(
        select(OrderRow).where(
            OrderRow.id == order_id.strip().upper(),
            OrderRow.customer_id == customer_id,
            OrderRow.store_id == store_id,
        )
    ).one_or_none()


SAMPLE_ORDERS: list[dict[str, str | None]] = [
    {"id": "ORD-1001", "store_id": "main-street-market", "customer_id": "cust-1001",
     "status": "delivered", "carrier": "UPS", "eta": None},
    {"id": "ORD-1042", "store_id": "main-street-market", "customer_id": "cust-1001",
     "status": "in_transit", "carrier": "FedEx", "eta": "2026-10-03"},
    {"id": "ORD-2003", "store_id": "main-street-market", "customer_id": "cust-2002",
     "status": "processing", "carrier": None, "eta": "2026-10-06"},
    {"id": "PNP-5001", "store_id": "pine-and-peak", "customer_id": "cust-3003",
     "status": "ready_for_pickup", "carrier": None, "eta": None},
]


def seed_orders(session: Session) -> None:
    """Upserts SAMPLE_ORDERS. Doesn't commit — the caller owns the transaction."""
    statement = insert(OrderRow).values(SAMPLE_ORDERS)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[OrderRow.id],
            set_={column: statement.excluded[column] for column in SAMPLE_ORDERS[0] if column != "id"},
        )
    )


def main() -> None:
    with new_session() as session:
        seed_orders(session)
        session.commit()
    print(f"orders: {len(SAMPLE_ORDERS)} sample orders seeded")


if __name__ == "__main__":
    main()
