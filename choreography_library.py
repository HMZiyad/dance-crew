"""
choreography_library.py
========================
Genre-specific choreographic move descriptions for Veo prompt injection.
Each entry describes SPECIFIC, RECOGNISABLE moves for that dance style so that
Veo generates real choreography rather than generic body sway.
"""

CHOREO = {
    "hip-hop": (
        "executing precise hip-hop choreography: stepping side-to-side with sharp arm isolations, "
        "hitting a hard freeze, popping chest on every beat, body wave rippling head to toe, "
        "crisp two-step footwork with heel-toe alt, wrist twists on the offbeat, "
        "sudden low crouch squat then spring to full height, rhythmic head bounce"
    ),
    "breakdancing": (
        "performing b-boy breaking: toprock front-to-back steps, drop into 6-step footwork, "
        "baby freeze on one hand, windmill spin on the floor, "
        "kip-up back to standing, threading legs through hands in a continuous flow, "
        "shoulder freeze into a headstand, flare leg circles"
    ),
    "breakdancing / b-boy": (
        "performing b-boy breaking: toprock front-to-back steps, drop into 6-step footwork, "
        "baby freeze on one hand, windmill spin on the floor, "
        "kip-up back to standing, flare leg circles, headstand freeze"
    ),
    "popping & locking": (
        "popping and locking: hitting sharp muscle contractions that travel from fingertips up arms, "
        "robot-like joint locks freezing mid-motion, wrist rolls, "
        "slow dimestop then instant pop explosion, tutting box shapes with arms, "
        "neck isolation sliding side to side, chest pop then stomach contraction wave"
    ),
    "popping and locking": (
        "popping and locking: sharp muscle contractions travelling fingertip to shoulder, "
        "robot joint locks, wrist rolls, dimestops, tutting arm boxes, chest pops"
    ),
    "krump": (
        "krumping: explosive full-body stomp landing flat-footed, heavy chest pound, "
        "aggressive arm swings from the shoulder, chest pop thrust forward, "
        "stomping forward with arms flailing, drop to a knees-on-floor grunt, "
        "spin up releasing tension, aggressive buck facial expression with full-body shudder"
    ),
    "waacking": (
        "waacking: rapid arm circles overhead in wide arcs, "
        "sharp whipping extension of arm straight out on the beat, "
        "pose hold with extended arm then sudden whip, "
        "spinning pivot with arm tracing a circle, side lunge into an arm pose, "
        "double waack alternating arms, hip push on each accent"
    ),
    "vogueing": (
        "vogueing: dip to the floor with one leg extended, "
        "catwalk runway strut with shoulder rolls, "
        "death drop falling back to the floor and snapping up, "
        "hand performance framing face, split hands framing silhouette, "
        "catwalk turn with hand on hip, duck walk low to the ground"
    ),
    "contemporary": (
        "contemporary dance: fluid contraction and release of the torso, "
        "sweeping leg extension arabesque, floor spiral roll recovering to standing, "
        "off-balance tilts catching weight on one foot, "
        "slow adagio arm sweep overhead, sharp fall to the ground and controlled rise, "
        "triplet step pattern across the floor, suspended balance on relevé"
    ),
    "ballet": (
        "classical ballet: relevé passé arabesque balance, "
        "grand battement leg swing to 90 degrees, chassé side travel, "
        "pirouette spin en dehors with clean finish fifth position, "
        "plié demi then grand in second, port de bras arm sweep, "
        "tendu point front side back, jeté leap with full leg extension"
    ),
    "salsa": (
        "salsa dancing: quick-quick-slow basic step forward and back, "
        "right turn with hand-led spin, cross-body lead stepping across, "
        "shoulder shimmies while stepping, cumbia rock side step, "
        "suzy Q hip rotation, inside turn back to frame, "
        "step tap accent with hip flick on count 2 and 6"
    ),
    "bachata": (
        "bachata: three-step side-to-side with hip tap on the fourth count, "
        "body wave flowing hip to chest, step touch with hip isolation, "
        "right and left turn spotting, dip with partner arm extension, "
        "syncopated double-time footwork, chest roll melting into hip"
    ),
    "twerk": (
        "twerking: rapid isolated glute oscillation, "
        "low squat position bouncing controlled, "
        "single-side hip pop alternating left and right, "
        "wall twerk hands on thighs, arch then tuck spine moving hips, "
        "standing bent-knee bounce with rolling hip drops"
    ),
    "k-pop": (
        "K-pop choreography: precise formation step with sharp arm point on the beat, "
        "synchronized group wave, hat-tip hand gesture, "
        "step-touch grapevine travel with body roll, "
        "clean shoulder bounce leading into a full-body roll, "
        "finger-gun pose on chorus hit, hip-sway march step, "
        "quick 180 turn ending in a strong pose facing camera"
    ),
    "bollywood": (
        "Bollywood choreography: expressive hasta hand gesture extended forward, "
        "thumka hip drop with hand frame by the ear, "
        "step-step-clap pattern advancing forward, "
        "circle arm bhangra raise with shoulder bounce, "
        "graceful neck tilts side to side, mudra hand to heart gesture, "
        "spin with dupatta-style arm float, balle-balle shoulder shrug"
    ),
    "afrobeats": (
        "Afrobeats choreography: shaku shaku shuffle step, "
        "zanku leg lift with arm pump, "
        "gwara gwara leaning dip footwork, "
        "azonto alternating arm and leg swing, "
        "shoulder shimmy rolling into chest isolation, "
        "drop low bounce then chest-up pop, "
        "two-step with hip swing on every bar"
    ),
    "jazz funk": (
        "jazz funk choreography: stylized jazz square with body roll, "
        "fan kick leg swing followed by a pop, "
        "hip thrust on the downbeat, "
        "finger-snap walk with shoulder leading, "
        "hair whip with full-body roll, "
        "split squat lunge into a spin, "
        "diva hand frame to the face, sharp layout arch backward"
    ),
    "tutting": (
        "tutting: precise 90-degree right-angle arm configurations changing on every beat, "
        "box shape morphing one joint at a time, "
        "wave travelling through sequential finger-wrist-elbow-shoulder joints, "
        "pyramid hand shape to flat frame, "
        "double-jointed box construction bilateral symmetry, "
        "slow joint-by-joint deconstruction then snap rebuild"
    ),
    "locking": (
        "funk locking: lock freeze mid-reach pointing index finger, "
        "wrist roll into another lock, "
        "scooby walk forward hopping on alternate feet, "
        "double lock clapping overheads, "
        "seated knee drop to the floor then spring up, "
        "knee drop with two-handed point, "
        "fast wrist circles into a full stop lock"
    ),
}


def get_choreo(dance_style: str) -> str:
    """Return the choreography description for a given dance style (case-insensitive)."""
    key = dance_style.lower().strip()
    # Exact match first
    if key in CHOREO:
        return CHOREO[key]
    # Partial match
    for k, v in CHOREO.items():
        if k in key or key in k:
            return v
    # Generic fallback
    return (
        f"performing authentic {dance_style} choreography with recognisable genre-specific footwork, "
        f"precise arm movements, rhythmic body isolations, and dynamic level changes — "
        f"full expert-level {dance_style} technique visible from head to toe"
    )
