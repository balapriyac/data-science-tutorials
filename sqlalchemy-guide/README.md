# SQLAlchemy ORM for Python Developers

## What the Article Covers

- Defining models with `Mapped` type annotations 
- One-to-many relationships with `relationship()` and `back_populates`
- Cascade delete with `cascade="all, delete-orphan"`
- Managing sessions with a context manager
- Inserting related records in a single transaction
- Filtering, joining, updating, and deleting rows
- Avoiding the N+1 query problem with `selectinload`

All examples use SQLite, which requires no setup beyond installing SQLAlchemy:

```bash
pip install sqlalchemy
```
