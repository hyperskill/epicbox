import tempfile

import structlog
import structlog._config


__all__ = ['Profile', 'configure']


IS_CONFIGURED = False
PROFILES = {}
DOCKER_URL = None
DOCKER_WORKDIR = '/sandbox'
BASE_WORKDIR = None

DEFAULT_LIMITS = {
    # CPU time in seconds, None for unlimited
    'cputime': 1,
    # Real time in seconds, None for unlimited
    'realtime': 5,
    # Memory in megabytes, None for unlimited
    'memory': 64,

    # Allow user process to fork
    #'canfork': False,
    # Limiting the maximum number of user processes in Linux is tricky.
    # http://unix.stackexchange.com/questions/55319/are-limits-conf-values-applied-on-a-per-process-basis
}
DEFAULT_USER = 'root'
CPU_TO_REAL_TIME_FACTOR = 5


class Profile(object):
    def __init__(self, name, docker_image, command=None, user=DEFAULT_USER,
                 read_only=False, network_disabled=True):
        self.name = name
        self.docker_image = docker_image
        self.command = command
        self.user = user
        self.read_only = read_only
        self.network_disabled = network_disabled


def configure(profiles=None, docker_url=None, base_workdir=None):
    global IS_CONFIGURED, PROFILES, DOCKER_URL, BASE_WORKDIR

    IS_CONFIGURED = True
    if isinstance(profiles, dict):
        PROFILES = {name: Profile(name, **profile_kwargs)
                    for name, profile_kwargs in profiles.items()}
    else:
        PROFILES = {profile.name: profile for profile in profiles or []}
    DOCKER_URL = docker_url
    if base_workdir is not None:
        BASE_WORKDIR = base_workdir
    else:
        BASE_WORKDIR = tempfile.gettempdir()


if not structlog._config._CONFIG.is_configured:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.KeyValueRenderer(key_order=['event']),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
