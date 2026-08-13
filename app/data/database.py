"""Database connection, ORM models and helper functions.

We use PostgreSQL through SQLAlchemy. The connection string comes from
config, which reads it from the .env file. Tests point the module at an
ephemeral database so the suite stays isolated. The models are simple
classes that map to tables:

- prices: historical OHLCV rows for each symbol.
- volatility_forecasts: predictions made by the models.
- backtest_runs: one row per backtest, with metrics stored as JSON.
- backtest_trades: individual trades produced by a backtest run.

The generator iter_price_chunks() streams prices in chunks instead of
loading everything at once, which is the generators requirement.
"""

from datetime import date, datetime
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, selectinload, sessionmaker

from app.config import DATABASE_URL
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Common base class for all ORM models."""


class Price(Base):
    """A single OHLCV bar for one symbol on one date."""

    __tablename__ = "prices"
    __table_args__ = (
        # One row per symbol and date, no duplicates allowed.
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
    )

    id: int = Column(Integer, primary_key=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    date: date = Column(Date, nullable=False, index=True)
    open: float = Column(Float, nullable=False)
    high: float = Column(Float, nullable=False)
    low: float = Column(Float, nullable=False)
    close: float = Column(Float, nullable=False)
    volume: float = Column(Float, default=0.0)
    source: str = Column(String(20), default="yahoo")

    def to_dict(self) -> Dict[str, Any]:
        """Return the row as a plain dict, safe for JSON responses."""
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source,
        }


class VolatilityForecast(Base):
    """A volatility prediction made by one of the models."""

    __tablename__ = "volatility_forecasts"

    id: int = Column(Integer, primary_key=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    model_name: str = Column(String(50), nullable=False)
    horizon_days: int = Column(Integer, default=5)
    forecast_date: date = Column(Date, nullable=False)
    predicted_volatility: float = Column(Float, nullable=False)
    actual_volatility: float = Column(Float, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Return the row as a plain dict, safe for JSON responses."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "model_name": self.model_name,
            "horizon_days": self.horizon_days,
            "forecast_date": self.forecast_date.isoformat(),
            "predicted_volatility": self.predicted_volatility,
            "actual_volatility": self.actual_volatility,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestRun(Base):
    """One backtest: the strategy, its params, and the results."""

    __tablename__ = "backtest_runs"

    id: int = Column(Integer, primary_key=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    strategy_name: str = Column(String(50), nullable=False)
    params: dict = Column(JSON, default=dict)
    metrics: dict = Column(JSON, default=dict)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    trades = relationship(
        "BacktestTrade",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return the row as a plain dict, safe for JSON responses."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "params": self.params,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestTrade(Base):
    """A single trade logged during a backtest run."""

    __tablename__ = "backtest_trades"

    id: int = Column(Integer, primary_key=True)
    run_id: int = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    date: date = Column(Date, nullable=False)
    action: str = Column(String(10), nullable=False)  # BUY or SELL
    price: float = Column(Float, nullable=False)
    size: int = Column(Integer, default=0)
    pnl: float = Column(Float, default=0.0)

    run = relationship("BacktestRun", back_populates="trades")

    def to_dict(self) -> Dict[str, Any]:
        """Return the row as a plain dict, safe for JSON responses."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "date": self.date.isoformat(),
            "action": self.action,
            "price": self.price,
            "size": self.size,
            "pnl": self.pnl,
        }


# Build the engine and session factory once. The DATABASE_URL comes from
# config, which reads it from .env.
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_database(url: str) -> None:
    """Point the database module at a different URL.

    Tests use this to run against an isolated database, and the app uses
    it to honor a custom DATABASE_URL from the environment.

    Args:
        url: a SQLAlchemy database URL.
    """
    global engine, SessionLocal
    engine = create_engine(url, echo=False)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    init_db()
    logger.info("Configured database at %s", url)


def init_db() -> None:
    """Create all tables if they do not exist yet.

    Safe to call on every startup.
    """
    Base.metadata.create_all(engine)
    logger.info("Database tables ready at %s", engine.url)


def get_session() -> Session:
    """Open a new session.

    Returns:
        A fresh Session bound to the engine.
    """
    return SessionLocal()


def save_prices(symbol: str, df: pd.DataFrame, source: str = "yahoo") -> int:
    """Insert cleaned OHLCV rows for a symbol, skipping duplicates.

    Args:
        symbol: the ticker.
        df: cleaned OHLCV data indexed by date.
        source: where the data came from, yahoo or csv.

    Returns:
        Number of rows actually inserted.
    """
    symbol = symbol.upper()
    inserted = 0
    with SessionLocal() as session:
        for index, row in df.iterrows():
            # Skip dates we already have for this symbol.
            exists = (
                session.query(Price)
                .filter(Price.symbol == symbol, Price.date == index.date())
                .first()
            )
            if exists:
                continue
            session.add(
                Price(
                    symbol=symbol,
                    date=index.date(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                    source=source,
                )
            )
            inserted += 1
        session.commit()
    logger.info("Saved %d new price rows for %s", inserted, symbol)
    return inserted


def load_prices(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load stored prices for a symbol into a DataFrame.

    Args:
        symbol: the ticker.
        start: optional start date YYYY-MM-DD.
        end: optional end date YYYY-MM-DD.

    Returns:
        OHLCV data indexed by date, or an empty frame if none found.
    """
    symbol = symbol.upper()
    with SessionLocal() as session:
        query = session.query(Price).filter(Price.symbol == symbol)
        if start:
            query = query.filter(Price.date >= date.fromisoformat(start))
        if end:
            query = query.filter(Price.date <= date.fromisoformat(end))
        rows = query.order_by(Price.date).all()

    data = {"open": [], "high": [], "low": [], "close": [], "volume": [], "date": []}
    for row in rows:
        data["open"].append(row.open)
        data["high"].append(row.high)
        data["low"].append(row.low)
        data["close"].append(row.close)
        data["volume"].append(row.volume)
        data["date"].append(row.date)

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def iter_price_chunks(symbol: str, chunksize: int = 500) -> Generator[List[Price], None, None]:
    """Stream stored prices for a symbol in chunks.

    A generator that yields lists of Price objects. Callers can process
    each chunk and forget about it, so memory use stays flat even for
    very long histories.

    Args:
        symbol: the ticker.
        chunksize: number of rows per chunk.

    Yields:
        Lists of Price objects.
    """
    symbol = symbol.upper()
    with SessionLocal() as session:
        query = (
            session.query(Price)
            .filter(Price.symbol == symbol)
            .order_by(Price.date)
            .yield_per(chunksize)
        )
        chunk = []
        for row in query:
            chunk.append(row)
            if len(chunk) >= chunksize:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def save_forecast(
    symbol: str,
    model_name: str,
    horizon_days: int,
    forecast_date: date,
    predicted_volatility: float,
    actual_volatility: Optional[float] = None,
) -> int:
    """Store one volatility forecast.

    Returns:
        The id of the new row.
    """
    with SessionLocal() as session:
        row = VolatilityForecast(
            symbol=symbol.upper(),
            model_name=model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            predicted_volatility=float(predicted_volatility),
            actual_volatility=actual_volatility,
        )
        session.add(row)
        session.commit()
        return row.id


def load_forecasts(symbol: str, model_name: Optional[str] = None) -> List[VolatilityForecast]:
    """Load stored forecasts for a symbol.

    Returns:
        List of VolatilityForecast rows, newest first.
    """
    with SessionLocal() as session:
        query = session.query(VolatilityForecast).filter(
            VolatilityForecast.symbol == symbol.upper()
        )
        if model_name:
            query = query.filter(VolatilityForecast.model_name == model_name)
        return query.order_by(VolatilityForecast.forecast_date.desc()).all()


def save_backtest(
    symbol: str,
    strategy_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    trades: List[Dict[str, Any]],
) -> int:
    """Store a finished backtest run and its trades.

    Returns:
        The id of the new run.
    """
    with SessionLocal() as session:
        run = BacktestRun(
            symbol=symbol.upper(),
            strategy_name=strategy_name,
            params=params,
            metrics=metrics,
        )
        session.add(run)
        session.flush()  # get the run id before adding trades

        for trade in trades:
            # Trades come in with ISO date strings, the Date column wants
            # real date objects so this also works on SQLite.
            trade_date = trade["date"]
            if isinstance(trade_date, str):
                trade_date = date.fromisoformat(trade_date[:10])

            session.add(
                BacktestTrade(
                    run_id=run.id,
                    date=trade_date,
                    action=trade["action"],
                    price=trade["price"],
                    size=trade.get("size", 0),
                    pnl=trade.get("pnl", 0.0),
                )
            )
        session.commit()
        return run.id


def load_backtest_runs(symbol: Optional[str] = None) -> List[BacktestRun]:
    """Load backtest runs, optionally filtered by symbol.

    Returns:
        List of BacktestRun rows, newest first.
    """
    with SessionLocal() as session:
        query = session.query(BacktestRun)
        if symbol:
            query = query.filter(BacktestRun.symbol == symbol.upper())
        return query.order_by(BacktestRun.created_at.desc()).all()


def load_backtest_run(run_id: int) -> Optional[BacktestRun]:
    """Load a single backtest run by id.

    Trades are eagerly loaded while the session is open, otherwise lazy
    loading would fail once the session is closed.

    Returns:
        The BacktestRun or None if it does not exist.
    """
    with SessionLocal() as session:
        return (
            session.query(BacktestRun)
            .options(selectinload(BacktestRun.trades))
            .filter(BacktestRun.id == run_id)
            .first()
        )
