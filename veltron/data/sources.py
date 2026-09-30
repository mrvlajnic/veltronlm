"""Dataset source registry: license, provenance and access metadata.

Every corpus VeltronLM trains on is declared here *before* it is downloaded. A source
that is not in this registry cannot be fetched by the acquisition pipeline, which makes
"what licence is this text under?" answerable by inspection rather than by guesswork.

Licence column values are the actual SPDX identifiers returned by the upstream projects
or the Wikimedia terms of use, not a judgement call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Source:
    """One fetchable corpus."""

    key: str
    name: str
    url_template: str
    license: str
    license_url: str
    attribution_required: bool
    commercial_use: str  # "permitted" | "restricted" | "unknown"
    redistribution: str  # "permitted" | "share-alike" | "restricted"
    category: str  # general | code | technical | instruction | support | multilingual
    languages: tuple[str, ...]
    notes: str = ""
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------ registry
SOURCES: dict[str, Source] = {}


def _reg(s: Source) -> Source:
    SOURCES[s.key] = s
    return s


# General knowledge -----------------------------------------------------------
_reg(Source(
    key="wikipedia-en",
    name="English Wikipedia (plain text extracts)",
    url_template="https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
                 "&explaintext=1&exlimit=20&redirects=1&formatversion=2&titles={titles}",
    license="CC-BY-SA-4.0",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="share-alike",
    category="general",
    languages=("en",),
    notes="Attribution and share-alike apply to redistributed derivatives of the text.",
))
_reg(Source(
    key="wikipedia-sr",
    name="Serbian Wikipedia (Latin and Cyrillic script editions)",
    url_template="https://{lang}.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
                 "&explaintext=1&exlimit=20&redirects=1&formatversion=2&titles={titles}",
    license="CC-BY-SA-4.0",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="share-alike",
    category="multilingual",
    languages=("sr", "sr-Latn", "sr-Cyrl", "sh", "hr", "bs"),
    notes="Primary source of Serbian-domain pretraining text.",
))
_reg(Source(
    key="wikipedia-random",
    name="Random-article sweep across Wikimedia projects",
    url_template="https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
                 "&generator=random&grnnamespace=0&grnlimit={count}&prop=extracts"
                 "&explaintext=1&exlimit=20&formatversion=2",
    license="CC-BY-SA-4.0",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="share-alike",
    category="general",
    languages=("en", "sr"),
    notes=(
        "Random-article generator with a fixed seed offset per run. The seed is recorded in "
        "the manifest so a rebuild can reproduce the exact article set."
    ),
))
_reg(Source(
    key="wikisource",
    name="Wikisource public-domain literary texts",
    url_template="https://{lang}.wikisource.org/w/api.php?action=query&format=json"
                 "&prop=extracts&explaintext=1&exlimit=20&redirects=1&formatversion=2&titles={titles}",
    license="CC-BY-SA-4.0 / public-domain source texts",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="share-alike",
    category="general",
    languages=("en", "sr"),
))

# Literature ------------------------------------------------------------------
_reg(Source(
    key="gutenberg",
    name="Project Gutenberg plain-text library",
    url_template="https://www.gutenberg.org/files/{id}/{id}-0.txt",
    license="PublicDomain",
    license_url="https://www.gutenberg.org/policy/license.html",
    attribution_required=False,
    commercial_use="permitted",
    redistribution="permitted",
    category="general",
    languages=("en",),
    notes="Texts are in the public domain in the US; check local law for older EU texts.",
))

# Code ------------------------------------------------------------------------
_reg(Source(
    key="cpython-stdlib",
    name="CPython standard library source",
    url_template="https://raw.githubusercontent.com/python/cpython/{ref}/Lib/{path}",
    license="PSF-2.0",
    license_url="https://github.com/python/cpython/blob/main/LICENSE",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="permitted",
    category="code",
    languages=("en",),
))
_reg(Source(
    key="permissive-repos",
    name="Curated MIT/Apache/BSD repositories",
    url_template="https://raw.githubusercontent.com/{repo}/{ref}/{path}",
    license="MIT OR Apache-2.0 OR BSD-3-Clause (per repository)",
    license_url="https://spdx.org/licenses/MIT.html",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="permitted",
    category="code",
    languages=("en",),
    notes="Each entry in PERMISSIVE_REPOS carries its own SPDX id; see the module constant.",
))

# Technical documentation -------------------------------------------------------
_reg(Source(
    key="pydoc",
    name="Python standard-library docstrings (technical English)",
    url_template="https://raw.githubusercontent.com/python/cpython/{ref}/Lib/{path}",
    license="PSF-2.0",
    license_url="https://github.com/python/cpython/blob/main/LICENSE",
    attribution_required=True,
    commercial_use="permitted",
    redistribution="permitted",
    category="technical",
    languages=("en",),
))

# ------------------------------------------------------------------------------ code corpus
@dataclass(frozen=True)
class RepoFile:
    repo: str
    ref: str
    path: str
    spdx: str
    role: str = "code"


#: Files pulled from permissively licensed projects. Each carries its SPDX identifier so
#: the licence of every downloaded byte is known individually.
CPYTHON_FILES: tuple[str, ...] = tuple(
    f"Lib/{p}"
    for p in (
        "abc.py", "argparse.py", "ast.py", "asyncio/base_events.py", "asyncio/tasks.py",
        "asyncio/queues.py", "asyncio/events.py", "base64.py", "bisect.py", "calendar.py",
        "collections/__init__.py", "collections/deque.py", "collections/abc.py",
        "concurrent/futures/_base.py", "configparser.py", "contextlib.py", "copy.py",
        "csv.py", "dataclasses.py", "datetime.py", "decimal.py", "difflib.py",
        "dis.py", "email/message.py", "email/parser.py", "enum.py", "fnmatch.py",
        "fractions.py", "functools.py", "gzip.py", "hashlib.py", "heapq.py",
        "hmac.py", "html/parser.py", "http/client.py", "http/server.py", "imaplib.py",
        "inspect.py", "io.py", "ipaddress.py", "itertools.py", "json/encoder.py",
        "json/decoder.py", "json/scanner.py", "logging/__init__.py", "logging/handlers.py",
        "logging/config.py", "lzma.py", "mimetypes.py", "multiprocessing/queues.py",
        "netrc.py", "numbers.py", "operator.py", "optparse.py", "os.py", "pathlib.py",
        "pdb.py", "pickle.py", "pkgutil.py", "platform.py", "plistlib.py", "poplib.py",
        "pprint.py", "profile.py", "pstats.py", "pty.py", "queue.py", "quopri.py",
        "random.py", "re/__init__.py", "re/_compiler.py", "reprlib.py", "resource.py",
        "secrets.py", "select.py", "shelve.py", "shlex.py", "shutil.py", "signal.py",
        "site.py", "smtplib.py", "socket.py", "socketserver.py", "sqlite3/dbapi2.py",
        "ssl.py", "stat.py", "statistics.py", "string.py", "stringprep.py",
        "struct.py", "subprocess.py", "symtable.py", "sysconfig.py", "tarfile.py",
        "tempfile.py", "textwrap.py", "threading.py", "timeit.py", "token.py",
        "tokenize.py", "traceback.py", "types.py", "typing.py", "unicodedata.py",
        "unittest/case.py", "unittest/loader.py", "unittest/mock.py", "urllib/parse.py",
        "urllib/request.py", "urllib/robotparser.py", "uuid.py", "venv/__init__.py",
        "warnings.py", "wave.py", "weakref.py", "webbrowser.py",
        "xml/etree/ElementTree.py", "xml/sax/expatreader.py", "xml/dom/minidom.py",
        "zipfile.py", "zlib.py", "zoneinfo/__init__.py",
    )
)

FLASK_CODE: tuple[str, ...] = (
    "src/flask/app.py", "src/flask/blueprints.py", "src/flask/config.py",
    "src/flask/helpers.py", "src/flask/ctx.py", "src/flask/sessions.py",
    "src/flask/wrappers.py", "src/flask/json/__init__.py", "src/flask/cli.py",
    "src/flask/templating.py", "src/flask/testing.py", "src/flask/globals.py",
)
FLASK_DOCS: tuple[str, ...] = (
    "docs/index.rst", "docs/api.rst", "docs/config.rst", "docs/patterns.rst",
    "docs/quickstart.rst", "docs/tutorial.rst", "docs/async.rst", "docs/server.rst",
)

#: Files pulled from permissively licensed projects. Each carries its SPDX identifier so
#: the licence of every downloaded byte is known individually.
PERMISSIVE_REPOS: tuple[RepoFile, ...] = (
    tuple(RepoFile("python/cpython", "3.12", p, "PSF-2.0") for p in CPYTHON_FILES)
    + tuple(RepoFile("pallets/flask", "main", p, "BSD-3-Clause") for p in FLASK_CODE)
    + tuple(RepoFile("pallets/flask", "main", p, "BSD-3-Clause", "technical") for p in FLASK_DOCS)
    + tuple(
        RepoFile("pallets/click", "main", p, "BSD-3-Clause")
        for p in ("src/click/core.py", "src/click/decorators.py", "src/click/types.py",
                  "src/click/parser.py", "src/click/termui.py", "src/click/utils.py",
                  "src/click/exceptions.py", "src/click/formatting.py")
    )
    + tuple(
        RepoFile("pallets/werkzeug", "main", p, "BSD-3-Clause")
        for p in ("src/werkzeug/routing.py", "src/werkzeug/wrappers/request.py",
                  "src/werkzeug/wrappers/response.py", "src/werkzeug/serving.py",
                  "src/werkzeug/security.py", "src/werkzeug/datastructures.py",
                  "src/werkzeug/utils.py")
    )
    + tuple(
        RepoFile("psf/requests", "main", p, "Apache-2.0")
        for p in ("requests/api.py", "requests/sessions.py", "requests/models.py",
                  "requests/adapters.py", "requests/auth.py", "requests/exceptions.py",
                  "requests/structures.py", "requests/utils.py", "requests/certs.py")
    )
    + tuple(
        RepoFile("psf/urllib3", "main", p, "MIT")
        for p in ("src/urllib3/connectionpool.py", "src/urllib3/connection.py",
                  "src/urllib3/response.py", "src/urllib3/util/retry.py",
                  "src/urllib3/poolmanager.py", "src/urllib3/exceptions.py",
                  "src/urllib3/util/ssl_.py", "src/urllib3/util/request.py")
    )
    + tuple(
        RepoFile("numpy/numpy", "main", p, "BSD-3-Clause", "technical")
        for p in ("README.md", "numpy/lib/_function_base_impl.py",
                  "numpy/lib/_shape_base_impl.py", "numpy/typing/_array_like.py")
    )
    + tuple(
        RepoFile("rust-lang/rust", "master", p, "MIT OR Apache-2.0", "technical")
        for p in ("README.md", "CONTRIBUTING.md", "SECURITY.md",
                  "CODE_OF_CONDUCT.md", "RELEASES.md")
    )
    + tuple(
        RepoFile("nodejs/node", "main", p, "MIT", "technical")
        for p in ("README.md", "SECURITY.md")
    )
    + tuple(
        RepoFile("golang/go", "master", p, "BSD-3-Clause", "technical")
        for p in ("README.md", "SECURITY.md", "CONTRIBUTING.md")
    )
    + tuple(
        RepoFile("psf/black", "main", p, "MIT", "technical")
        for p in ("README.md", "docs/the_black_code_style/current_style.md")
    )
    + tuple(
        RepoFile("pallets/werkzeug", "main", p, "BSD-3-Clause", "technical")
        for p in ("CHANGES.rst", "docs/index.rst", "docs/quickstart.rst", "docs/wrappers.rst")
    )
)

# Wikipedia article title lists. Kept explicit so the acquisition is reproducible and
# does not depend on a random article generator.
WIKIPEDIA_TITLES: dict[str, tuple[str, ...]] = {
    "en": (
        "Machine learning", "Transformer (machine learning model)", "Neural network",
        "Computer science", "Mathematics", "Physics", "Chemistry", "Biology",
        "History", "Geography", "Economics", "Psychology", "Philosophy", "Software engineering",
        "Computer programming", "Python (programming language)", "C (programming language)",
        "Rust (programming language)", "Operating system", "Database", "Internet",
        "Artificial intelligence", "Statistics", "Linear algebra", "Calculus",
        "Roman Empire", "Byzantine Empire", "Serbia", "Belgrade", "Europe", "World War II",
        "Climate", "Ecology", "Astronomy", "Medicine", "Law", "Literature", "Music",
        "Linguistics", "Serbian language", "Serbian orthography", "South Slavic languages",
        "Informatics", "Data", "Algorithm", "Complexity theory", "Graph theory",
        "Electronics", "Telecommunications", "Aerospace engineering", "Civil engineering",
    ),
    "sr": (
        "Вештачка интелигенција", "Машинско учење", "Нервне мреже",
        "Информатика", "Рачунарство", "Програмирање", "Питон", "Енглески језик",
        "Србија", "Београд", "Европа", "Историја", "Географија", "Математика",
        "Физика", "Хемија", "Биологија", "Економија", "Психологија", "Философија",
        "Српски језик", "Правопис српског језика", "Јужнословенски језици",
        "Вештачка интелигенција у медијима", "База података", "Оперативни систем",
        "Рачунарска мрежа", "Интернет", "Литература", "Музика", "Уметност",
        "Правo", "Медицина", "Астрономија", "Климатологија", "Екологија",
        "Држава", "Влада", "Политика", "Избори", "Економија",
    ),
}
WIKIPEDIA_SR_LATIN_TITLES: tuple[str, ...] = (
    "Veštačka inteligencija", "Mašinsko učenje", "Nervne mreže", "Informatika",
    "Računarstvo", "Programiranje", "Python (programski jezik)", "Engleski jezik",
    "Srbija", "Beograd", "Evropa", "Istorija", "Geografija", "Matematika",
    "Fizika", "Hemija", "Biologija", "Ekonomija", "Psihologija", "Filozofija",
    "Srpski jezik", "Pravopis srpskog jezika", "Južnoslovenski jezici",
    "Baza podataka", "Operativni sistem", "Računarska mreža", "Internet",
    "Literatura", "Muzika", "Umetnost", "Pravo", "Medicina", "Astronomija",
    "Klimatologija", "Ekologija", "Država", "Vlada", "Politika", "Izbori",
)


def enabled_sources(category: str | None = None) -> list[Source]:
    out = [s for s in SOURCES.values() if s.enabled]
    if category:
        out = [s for s in out if s.category == category]
    return out


def licence_report() -> list[dict[str, Any]]:
    return [
        {
            "key": s.key,
            "name": s.name,
            "license": s.license,
            "license_url": s.license_url,
            "attribution_required": s.attribution_required,
            "commercial_use": s.commercial_use,
            "redistribution": s.redistribution,
            "category": s.category,
            "languages": ",".join(s.languages),
            "enabled": s.enabled,
        }
        for s in SOURCES.values()
    ]


__all__ = ["Source", "RepoFile", "SOURCES", "PERMISSIVE_REPOS", "WIKIPEDIA_TITLES",
           "WIKIPEDIA_SR_LATIN_TITLES", "enabled_sources", "licence_report"]
