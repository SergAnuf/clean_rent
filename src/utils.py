from time import perf_counter

class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.start
        print(f"⏱ {self.label} took {elapsed:.2f}s")