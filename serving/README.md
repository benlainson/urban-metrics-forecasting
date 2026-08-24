# serving/

Shared FastAPI application that serves predictions from all three domain models behind one API. Owned jointly — each teammate adds their domain's endpoints/routers rather than forking the app.

See `URBAN_PULSE_PROJECT_PLAN.md` Phase 4 for the starter `main.py` and `Dockerfile`. Suggested convention: one router module per domain (e.g. `serving/routers/transit.py`) mounted into `main.py`, so merges don't collide on a single file.
