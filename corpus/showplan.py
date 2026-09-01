"""Plan a whole show: which episodes need transcription, embedding, neither.

WHY THIS IS NOT IN transcribe.py. Planning a show costs two Chroma round
trips per episode and re-chunks every transcript it finds -- roughly 870
round trips and 433 re-chunks per night across the three shows -- and none of
it needs a GPU. Run inside the `gpu="T4"` container it ran in originally,
that is minutes of accelerator time burned before any accelerator work
starts, on a schedule, every night, and usually to conclude there is nothing
to do at all. Spec 4.5 wrote the split explicitly and the first
implementation lost it.

Living here rather than in the Modal file buys the second thing: this module
imports no Modal, so the loop is reachable from the test suite. The
failure-isolation behaviour below -- one episode's transient Chroma error
must not abort the show -- previously had no test, only a comment claiming
it worked.

The filesystem is injected as `read_transcript` rather than opened here, so
the tests do not need a volume and this module does not need to know that a
Modal volume is what it is reading from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from corpus.completeness import plan_episode
from corpus.planning import Action


@dataclass
class ShowPlan:
    """The work a show needs, split by what kind of container can do it.

    `to_transcribe` and `to_embed` carry the feed's own episode dicts
    unchanged, because they cross a process boundary to the GPU worker and
    anything richer would have to survive serialisation for no gain. Ordering
    is the feed's ordering, preserved.
    """

    to_transcribe: list[dict] = field(default_factory=list)
    to_embed: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.to_transcribe or self.to_embed)


def plan_show(
    collection,
    *,
    show: str,
    episodes: list[dict],
    read_transcript: Callable[[dict], str | None],
    log: Callable[[str], None] = print,
) -> ShowPlan:
    """Decide what each episode of a show needs.

    `read_transcript` returns the transcript text for an episode, or None if
    the episode has no transcript on disk. It is allowed to raise: a volume
    read failing is treated exactly like a Chroma read failing, because from
    here they are the same event -- this episode could not be judged.

    Every episode is planned independently. A transient failure on one is
    recorded and the loop continues, because the alternative -- which is what
    the original did by leaving the call outside any try -- is that one bad
    Chroma response aborts an entire show before any work is done. The code
    this replaced used os.path.exists, which could not fail that way, so the
    regression arrived with the feature.
    """
    plan = ShowPlan()

    for episode in episodes:
        number = episode["episode_number"]
        date_str = episode["date"]
        try:
            action = plan_episode(
                collection,
                show=show,
                episode_number=number,
                date_str=date_str,
                transcript_text=read_transcript(episode),
                # Load-bearing: without it the exclusion check falls back to
                # the triple alone and the guid arm -- which exists precisely
                # because the triple is not durable -- never fires on the
                # nightly path.
                episode_guid=episode.get("guid"),
            )
        except Exception as e:
            log(
                f"  Failed to plan Episode {number} ({date_str}): "
                f"{type(e).__name__}: {e}"
            )
            plan.failures.append(f"{show} ep{number} ({date_str}) [plan]: {e}")
            continue

        log(f"  {action.value:12s} Episode {number} ({date_str})")

        if action is Action.TRANSCRIBE:
            plan.to_transcribe.append(episode)
        elif action is Action.EMBED_ONLY:
            plan.to_embed.append(episode)
        elif action is Action.SKIP:
            pass  # complete and current -- nothing to do
        elif action is Action.EXCLUDE:
            pass  # deliberately kept out -- see corpus/exclusions.py
        elif action is Action.UNPARSEABLE:
            pass  # transcript exists but yields no chunks -- terminal, since
            # treating it as incomplete would re-embed it forever
        else:
            # A silent no-op here has the same shape as the incident's
            # `except Exception: continue`: work goes unreported as undone.
            raise RuntimeError(f"unhandled Action from plan_episode: {action!r}")

    return plan
