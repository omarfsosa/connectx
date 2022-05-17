import nox

locations = "phil", "tests", "noxfile.py"
nox.options.sessions = "lint", "tests"


@nox.session(name="format")
def format_(session):
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@nox.session
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black", "isort")
    session.run("flake8", *args)
    session.run("isort", "--check", *args)


@nox.session
def tests(session):
    session.install("pytest", ".")
    session.run("pytest")
