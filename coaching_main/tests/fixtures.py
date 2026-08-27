"""Shared test fixture: a 40-turn coaching session with known properties.

Ground truth: 20 coach turns, 20 coachee turns, strictly alternating.
The session moves Goal -> Reality -> Options -> Way Forward. Turn 8 and
turn 10 are sarcastic; turn 16 contains an explicit topic digression.
"""

from types import SimpleNamespace
from typing import List, Tuple

TURNS: List[Tuple[str, str]] = [
    ("coach", "Hi Priya, good to see you. Before we dive in - what would you like to get out of our time today?"),
    ("coachee", "Hi Marcus. Honestly, I want to figure out why I keep getting passed over for the team-lead position. This is the third cycle now."),
    ("coach", "Okay. So if this session went really well, what would you walk away with?"),
    ("coachee", "A clear picture of what I'm actually doing wrong. I want to see it laid out, because right now it's just a blur."),
    ("coach", 'That\'s a good goal. Let\'s make it concrete - what does "getting the role" look like for you, specifically?'),
    ("coachee", "Leading the analytics pod. Owning the roadmap. Having people actually come to me for decisions instead of going around me."),
    ("coach", "Good. Let's hold that as the goal for today. Now tell me about the current situation - what's actually happening at work right now?"),
    ("coachee", 'So the last review, my manager said I "wasn\'t demonstrating leadership." Which is hilarious, because I\'m the one who stayed until midnight fixing the pipeline nobody else would touch.'),
    ("coach", "I hear the frustration in that. Say more about the gap between what you did and what he saw."),
    ("coachee", "I mean, I do the work. I just don't announce it. I assumed people would notice. Clearly that worked out great for me."),
    ("coach", "So there may be a difference between doing the work and making the work visible. Does that land for you?"),
    ("coachee", "...Yeah. It does, actually. I never really thought about it that way. It kind of stings to hear, but it sounds right."),
    ("coach", "Take a second with that. What comes up?"),
    ("coachee", "I guess I feel a bit exposed. Like maybe I've been telling myself a story where I'm the victim, and it's more complicated than that."),
    ("coach", 'That\'s a really honest thing to notice. Where else does this "doing but not showing" pattern show up?'),
    ("coachee", "Everywhere, now that I look at it. In meetings I stay quiet unless I'm a hundred percent sure. Oh - random, but did you watch the game last night? The overtime was unreal."),
    ("coach", "Ha, I caught the highlights. Let's park that though - I want to stay with what you just said about meetings."),
    ("coachee", "Right, sorry. Yeah. I hang back in meetings. I let louder people take the air."),
    ("coach", "So in the room, the version of you that solves the hard problems is basically invisible. How does that feel to say out loud?"),
    ("coachee", 'Kind of heavy. But also weirdly a relief? Like I can finally get a grip on something concrete instead of this vague "you\'re not leadership material" thing.'),
    ("coach", "Good - that's the shift from problem to something workable. So let's open it up. What are some options for making your contribution more visible?"),
    ("coachee", "Um. I could speak up earlier in meetings even when I'm not certain. I could send a short update on what my team shipped each week."),
    ("coach", "Keep going - don't filter yet. What else?"),
    ("coachee", 'I could ask my manager directly what "demonstrating leadership" means to him, instead of guessing. I could mentor one of the juniors, which would make the leadership thing visible. And maybe volunteer to present the roadmap next quarter.'),
    ("coach", "That's a strong list. Which of those actually excites you, versus feels like a chore?"),
    ("coachee", "Presenting the roadmap genuinely excites me. I can already picture standing up and walking everyone through it. The weekly update feels like a chore, honestly, but a small one."),
    ("coach", "Interesting that the biggest, most visible one is the one that excites you. What's underneath that?"),
    ("coachee", "I think I actually want to be seen. I've just been pretending I don't care, because caring and then not getting it hurts more."),
    ("coach", "That's a lot of self-awareness for one session. So of everything we've talked about, what feels like the right first step?"),
    ("coachee", "Booking a conversation with my manager to ask what leadership looks like to him. If I don't know the target, the rest is guesswork."),
    ("coach", "Love it. When will you do that by?"),
    ("coachee", "By Friday. I'll send the invite tomorrow morning, actually, before I lose my nerve."),
    ("coach", "And how will you know it went well?"),
    ("coachee", "If I walk out with two or three concrete things he'd need to see from me. Not vibes - specifics I can act on."),
    ("coach", "Perfect. On a scale of one to ten, how committed are you to sending that invite tomorrow?"),
    ("coachee", "A nine. The only reason it's not a ten is that mornings are chaos, but I'll do it."),
    ("coach", "What would make it a ten?"),
    ("coachee", "Setting a reminder for 9am and just doing it before I check email. Done, I'll set it now."),
    ("coach", "Beautiful. Let's recap: you noticed the real gap is visibility, not ability, and your first move is that manager conversation by Friday. How are you feeling now versus when we started?"),
    ("coachee", "Way lighter. I came in feeling stuck and kind of resentful, and I'm leaving with something I can actually see myself doing. Thank you, Marcus."),
]

#: 1-indexed turns that are genuinely sarcastic.
SARCASTIC_TURNS = (8, 10)

#: 1-indexed turn containing an explicit off-topic digression.
DIGRESSION_TURN = 16


def chunks(coach_label: str = "A"):
    """Build AudioChunk-like objects with diarization ids.

    ``coach_label`` selects which diarization label the coach receives -
    this is arbitrary in practice, and getting it wrong used to skew the
    speaker split badly.
    """
    other = "B" if coach_label == "A" else "A"
    return [
        SimpleNamespace(
            speaker=speaker,
            transcript=text,
            text=text,
            speaker_id=(coach_label if speaker == "coach" else other),
            timestamp=float(index * 8),
            duration=6.0,
            audio_data=None,
            is_final=True,
        )
        for index, (speaker, text) in enumerate(TURNS)
    ]


def utterances(coach_label: str = "A"):
    """Diarizer-shaped objects (``.speaker`` / ``.text``) for the router."""
    other = "B" if coach_label == "A" else "A"
    return [
        SimpleNamespace(
            speaker=(coach_label if speaker == "coach" else other), text=text
        )
        for speaker, text in TURNS
    ]
