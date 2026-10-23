import sqlite3
from contextlib import closing

import pytest

from jev_runtime import registry_backup, registry_migration


@pytest.fixture
def track_connections(monkeypatch):
    original = sqlite3.connect
    opened = []

    def track(factory=sqlite3.Connection):
        def connect(*args, **kwargs):
            connection = original(*args, **kwargs, factory=factory)
            opened.append(connection)
            return connection

        monkeypatch.setattr(registry_backup.sqlite3, "connect", connect)
        return opened

    yield track
    for connection in opened:
        connection.close()


def assert_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        sqlite3.Connection.execute(connection, "SELECT 1")


@pytest.mark.parametrize("writable", [False, True])
def test_corrupt_database_closes_connection_before_raising(tmp_path, track_connections, writable):
    source = tmp_path / "broken.db"
    content = b"not a sqlite database" * 100
    source.write_bytes(content)
    opened = track_connections()

    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        registry_backup.connect(source, writable=writable)

    assert len(opened) == 1
    assert_closed(opened[0])
    assert source.read_bytes() == content


@pytest.mark.parametrize("operation", ["snapshot", "inspect"])
def test_backup_tools_close_failed_source_before_publishing(tmp_path, track_connections, operation):
    source = tmp_path / "broken.db"
    source.write_bytes(b"not a sqlite database" * 100)
    destination = tmp_path / "snapshot"
    opened = track_connections()

    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        if operation == "snapshot":
            registry_backup.snapshot(source, destination)
        else:
            registry_migration.inspect(source)

    assert len(opened) == 1
    assert_closed(opened[0])
    assert not destination.exists()


@pytest.mark.parametrize(
    ("writable", "failing_statement"),
    [
        (False, "PRAGMA trusted_schema=OFF"),
        (False, "PRAGMA query_only=ON"),
        (True, "PRAGMA synchronous=FULL"),
    ],
)
@pytest.mark.parametrize("error_type", [sqlite3.OperationalError, KeyboardInterrupt])
def test_pragma_failure_or_interrupt_closes_connection(
    tmp_path, track_connections, writable, failing_statement, error_type
):
    source = tmp_path / "registry.db"
    source.touch()
    failure = error_type("injected initialization failure")

    class FailingConnection(sqlite3.Connection):
        def execute(self, statement, *args, **kwargs):
            if statement == failing_statement:
                raise failure
            return super().execute(statement, *args, **kwargs)

    opened = track_connections(FailingConnection)
    with pytest.raises(error_type) as caught:
        registry_backup.connect(source, writable=writable)

    assert caught.value is failure
    assert len(opened) == 1
    assert_closed(opened[0])


@pytest.mark.parametrize("writable", [False, True])
def test_successful_connection_remains_usable_with_requested_mode(tmp_path, writable):
    source = tmp_path / "registry.db"
    source.touch()

    with closing(registry_backup.connect(source, writable=writable)) as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        assert connection.execute("PRAGMA trusted_schema").fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA query_only").fetchone() == (int(not writable),)
        if writable:
            connection.execute("CREATE TABLE value (id INTEGER)")
            connection.commit()
        else:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("CREATE TABLE value (id INTEGER)")

    assert_closed(connection)
