class JevError(Exception):
    """An actionable, client-visible error without implementation tracebacks."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}
