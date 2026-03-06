import errno
import secrets
from datetime import datetime
from pathlib import Path
from types import TracebackType

from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class for type checking."""

    pass


class Subject(Base):
    """SQLAlchemy model representing an anonymized subject mapping."""

    __tablename__ = "subjects"

    anon_id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    patient_name: Mapped[str | None] = mapped_column(String, nullable=True)
    patient_birth_date: Mapped[str | None] = mapped_column(String, nullable=True)
    patient_sex: Mapped[str | None] = mapped_column(String, nullable=True)
    date_shift_days: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(default=datetime.now)

    sessions: Mapped[list["Session"]] = relationship("Session", back_populates="subject")


class Session(Base):
    """SQLAlchemy model representing a single conversion session for a subject."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    anon_id: Mapped[str] = mapped_column(String, ForeignKey("subjects.anon_id"), nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    original_study_uid: Mapped[str | None] = mapped_column(String, nullable=True)
    institution_name: Mapped[str | None] = mapped_column(String, nullable=True)
    converted_at: Mapped[datetime | None] = mapped_column(default=datetime.now)

    subject: Mapped["Subject"] = relationship("Subject", back_populates="sessions")


class AnonDB:
    """Interface to the SQLite anonymization mapping database."""

    def __init__(self, db_path: Path):
        """Open or create the database at the given path."""
        try:
            self.engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(self.engine)
            self._Session = sessionmaker(bind=self.engine)
            self.session = self._Session()
        except PermissionError as e:
            raise RuntimeError(f"Permission denied: cannot access database at {db_path}") from e
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise RuntimeError(f"Disk full: cannot create database at {db_path}") from e
            raise RuntimeError(f"Cannot open database at {db_path}: {e}") from e
        except OperationalError as e:
            raise RuntimeError(f"Database error at {db_path}: {e}") from e

    def __enter__(self) -> "AnonDB":
        """Support use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the database when exiting the context manager."""
        self.close()

    def get_or_create_subject(
        self,
        patient_id: str,
        patient_name: str | None = None,
        patient_birth_date: str | None = None,
        patient_sex: str | None = None,
        date_shift_days: int | None = None,
        anon_id: str | None = None,
    ) -> str:
        """Look up a subject by patient_id, or create a new one.

        If anon_id is provided it is used as the anonymous ID, otherwise a
        random one is generated. Returns the anon_id."""
        subject = self.session.query(Subject).filter_by(patient_id=patient_id).first()
        if subject is not None:
            return subject.anon_id

        if anon_id is None:
            # Generate unique anon_id (collision extremely unlikely with 64 bits)
            anon_id = secrets.token_hex(8)
            while self.session.query(Subject).filter_by(anon_id=anon_id).first() is not None:
                anon_id = secrets.token_hex(8)

        subject = Subject(
            anon_id=anon_id,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_birth_date=patient_birth_date,
            patient_sex=patient_sex,
            date_shift_days=date_shift_days,
        )
        self.session.add(subject)
        return anon_id

    def add_session(
        self,
        anon_id: str,
        source_path: str,
        original_study_uid: str | None = None,
        institution_name: str | None = None,
    ) -> str:
        """Record a new conversion session for a subject. The session_id is
        auto-incremented per subject. Returns the session_id string."""
        count = self.session.query(Session).filter_by(anon_id=anon_id).count()
        session_id = str(count + 1)
        sess = Session(
            anon_id=anon_id,
            session_id=session_id,
            source_path=source_path,
            original_study_uid=original_study_uid,
            institution_name=institution_name,
        )
        self.session.add(sess)
        return session_id

    def get_all_subjects(self) -> list[Subject]:
        """Return all subject records from the database."""
        return self.session.query(Subject).all()

    def get_sessions_for_subject(self, anon_id: str) -> list[Session]:
        """Return all session records for a given anonymous subject ID."""
        return self.session.query(Session).filter_by(anon_id=anon_id).all()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self.session.rollback()

    def close(self) -> None:
        """Close the database session and dispose of the engine."""
        self.session.close()
        self.engine.dispose()
