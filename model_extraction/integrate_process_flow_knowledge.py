import re

from pyspn.components.spn import SPN, Place, Transition


# ============================================================
# Metadata
# ============================================================

def _set_place_metadata(
    place,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    if has_conflict:
        place.color = "firebrick"
        place.conflicted_with = (
            "[!] -> "
            + str(id)
            + " vs. "
            + ", ".join(
                str(c["statement_id"])
                for c in conflicts
            )
        )
    else:
        place.color = color
        place.conflicted_with = conflicts

    place.has_conflict = has_conflict
    place.source = id


def _apply_place_metadata(
    place,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    _set_place_metadata(
        place,
        id,
        has_conflict,
        conflicts,
        color
    )


def _apply_transition_metadata(
    transition,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    transition.color = (
        "firebrick"
        if has_conflict
        else color
    )

    transition.has_conflict = has_conflict

    if has_conflict:
        transition.conflicted_with = (
            "[!] -> "
            + str(id)
            + " vs. "
            + ", ".join(
                str(c["statement_id"])
                for c in conflicts
            )
        )
    else:
        transition.conflicted_with = conflicts

    transition.source = id


# ============================================================
# API creation helpers
# ============================================================

def get_or_create_transition(
    spn,
    label,
    id,
    has_conflict,
    conflicts,
    t_type="I",
    color="forestgreen"
):
    label = str(label).strip()

    transition = spn.get_transition_by_label(label)

   

    if transition is not None:
        transition.has_conflict = has_conflict

        if has_conflict == True:
            transition.color = "firebrick"
            transition.source = id
            transition.conflicted_with = (
            "[!] -> "
            + str(id)
            + " vs. "
            + ", ".join(
                str(c["statement_id"])
                for c in conflicts
            )
            )

        return transition

    transition = Transition(
        label,
        t_type
    )

    _apply_transition_metadata(
        transition,
        id,
        has_conflict,
        conflicts,
        color
    )

    

    spn.add_transition(transition)

    return transition


def get_or_create_place(
    spn,
    label,
    tokens=0
):
    label = str(label).strip()

    place = spn.get_place_by_label(label)

    if place is not None:
        return place

    place = Place(
        label,
        tokens
    )

    spn.add_place(place)

    return place


# ============================================================
# Arc inspection
# ============================================================

def _has_output_arc(
    transition,
    place
):
    return any(
        arc.to_place is place
        for arc in transition.output_arcs
    )


def _has_input_arc(
    place,
    transition
):
    return any(
        arc.from_place is place
        for arc in transition.input_arcs
    )


def _has_transition_to_place_arc(
    transition,
    place
):
    return any(
        arc.from_transition is transition
        and arc.to_place is place
        for arc in transition.output_arcs
    )


def _has_place_to_transition_arc(
    place,
    transition
):
    return any(
        arc.from_place is place
        and arc.to_transition is transition
        for arc in transition.input_arcs
    )


# ============================================================
# Arc creation
# ============================================================

def _add_arc_place_to_transition(
    spn,
    place,
    transition,
    color="forestgreen"
):
    if not _has_place_to_transition_arc(
        place,
        transition
    ):
        spn.add_input_arc(
            place,
            transition,
            color=color
        )


def _add_arc_transition_to_place(
    spn,
    transition,
    place,
    color="forestgreen"
):
    if not _has_transition_to_place_arc(
        transition,
        place
    ):
        spn.add_output_arc(
            transition,
            place,
            color=color
        )
    else:
        transition.has_conflict == True
        transition.color = "firebrick"


def _connect_transition_place(
    spn,
    transition,
    place,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    _add_arc_transition_to_place(
        spn,
        transition,
        place,
        arc_color
    )

    return place


def _connect_place_transition(
    spn,
    place,
    transition,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    _add_arc_place_to_transition(
        spn,
        place,
        transition,
        arc_color
    )

    return place


# ============================================================
# Parsing
# ============================================================

def parse_if_then_complex(sentence):
    if sentence is None:
        return None, None

    sentence = str(sentence).strip()

    sentence = sentence.replace(
        "\u2010",
        "_"
    )

    match = re.match(
        r"^IF\s+(.+?)\s+THEN\s+(.+?)$",
        sentence,
        flags=re.IGNORECASE
    )

    if match is None:
        return None, None

    return (
        match.group(1).strip(),
        match.group(2).strip()
    )


def parse_expression(expression):
    if expression is None:
        return None, []

    expression = str(expression).strip()

    if re.search(
        r"\sAND\s",
        expression,
        flags=re.IGNORECASE
    ):
        activities = [
            x.strip()
            for x in re.split(
                r"\s+AND\s+",
                expression,
                flags=re.IGNORECASE
            )
            if x.strip()
        ]

        return "AND", activities

    if re.search(
        r"\sOR\s",
        expression,
        flags=re.IGNORECASE
    ):
        activities = [
            x.strip()
            for x in re.split(
                r"\s+OR\s+",
                expression,
                flags=re.IGNORECASE
            )
            if x.strip()
        ]

        return "OR", activities

    return "SINGLE", [expression]


# ============================================================
# Normal transition -> transition
#
# This remains the ordinary structure.
#
# A -> P -> B
#
# No XOR is introduced here.
# ============================================================

def connect_transition_to_transition(
    spn,
    source,
    target,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    place_label = (
        f"{source.label}_TO_{target.label}"
    )

    place = get_or_create_place(
        spn,
        place_label
    )

    _apply_place_metadata(
        place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _connect_transition_place(
        spn,
        source,
        place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _connect_place_transition(
        spn,
        place,
        target,
        id,
        has_conflict,
        conflicts,
        color
    )

    return place


# ============================================================
# Conditional bookkeeping
# ============================================================

def _conditional_state(spn):
    if not hasattr(
        spn,
        "_conditional_then_paths"
    ):
        spn._conditional_then_paths = {}

    return spn._conditional_then_paths


def _get_conditional_state(
    spn,
    then_transition
):
    state = _conditional_state(spn)

    key = then_transition.label

    if key not in state:
        state[key] = {
            "then": then_transition,
            "branches": [],
            "merge_transition": None,
            "output_place": None,
        }

    return state[key]


# ============================================================
# Conditional branch identity
# ============================================================

def _branch_key(
    condition_type,
    condition_labels,
    then_transition
):
    return (
        condition_type,
        tuple(condition_labels),
        then_transition.label
    )


def _branch_already_exists(
    state,
    key
):
    return any(
        branch["key"] == key
        for branch in state["branches"]
    )


# ============================================================
# Common conditional output
#
# Every conditional path ultimately becomes:
#
#       branch completion places
#                 |
#                 v
#          CONDITIONAL_XOR_JOIN
#                 |
#                 v
#                 P
#                 |
#                 v
#                 THEN
#
# The join is a TRANSITION, not a place.
#
# A transition with multiple input places can fire when
# any one of those places has a token, giving us the
# desired alternative merge.
# ============================================================

def _ensure_conditional_merge(
    spn,
    then_transition,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    state = _get_conditional_state(
        spn,
        then_transition
    )

    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    if state["merge_transition"] is None:

        merge_transition = get_or_create_transition(
            spn,
            (
                f"CONDITIONAL_"
                f"{then_transition.label}_"
                f"XOR_JOIN"
            ),
            id,
            has_conflict,
            conflicts,
            color=color
        )

        output_place = get_or_create_place(
            spn,
            (
                f"CONDITIONAL_"
                f"{then_transition.label}_"
                f"XOR_JOIN_OUTPUT"
            )
        )

        _apply_place_metadata(
            output_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            merge_transition,
            output_place,
            arc_color
        )

        _add_arc_place_to_transition(
            spn,
            output_place,
            then_transition,
            arc_color
        )

        state["merge_transition"] = merge_transition
        state["output_place"] = output_place

    return state


# ============================================================
# SINGLE conditional branch
#
# IF A THEN C
#
# A -> P -> conditional-XOR-JOIN -> P -> C
#
# IMPORTANT:
#
# This function is ONLY used when a conditional structure
# is actually required.
#
# Ordinary A -> C statements do NOT use this.
# ============================================================

def _add_single_conditional_branch(
    spn,
    source_transition,
    then_transition,
    branch_number,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    state = _ensure_conditional_merge(
        spn,
        then_transition,
        id,
        has_conflict,
        conflicts,
        color
    )

    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    completion_place = get_or_create_place(
        spn,
        (
            f"CONDITIONAL_"
            f"{then_transition.label}_"
            f"SINGLE_BRANCH_"
            f"{branch_number}_COMPLETION"
        )
    )

    _apply_place_metadata(
        completion_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _add_arc_transition_to_place(
        spn,
        source_transition,
        completion_place,
        arc_color
    )

    _add_arc_place_to_transition(
        spn,
        completion_place,
        state["merge_transition"],
        arc_color
    )

    return state


# ============================================================
# Genuine AND condition
#
# IF A AND B THEN C
#
# Correct structure:
#
#       A ----> P_A ----\
#                         \
#                          AND_JOIN -> P -> C
#                         /
#       B ----> P_B ----/
#
# There is NO artificial AND_SPLIT before A/B.
#
# There are also no unused branch input places.
# ============================================================

def _add_and_conditional_branch(
    spn,
    condition_transitions,
    then_transition,
    branch_number,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    state = _ensure_conditional_merge(
        spn,
        then_transition,
        id,
        has_conflict,
        conflicts,
        color
    )

    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    labels = [
        t.label
        for t in condition_transitions
    ]

    branch_prefix = (
        f"CONDITIONAL_"
        f"{then_transition.label}_"
        f"AND_BRANCH_{branch_number}"
    )

    # --------------------------------------------------------
    # One completion place per AND condition.
    #
    # These are the ONLY places introduced for the individual
    # condition completions.
    # --------------------------------------------------------

    completion_places = []

    for transition in condition_transitions:

        completion_place = get_or_create_place(
            spn,
            (
                f"{branch_prefix}_"
                f"{transition.label}_DONE"
            )
        )

        _apply_place_metadata(
            completion_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            transition,
            completion_place,
            arc_color
        )

        completion_places.append(
            completion_place
        )

    # --------------------------------------------------------
    # One AND JOIN transition.
    #
    # It has one input place for EVERY AND condition.
    #
    # Therefore all conditions must be available before
    # the transition can fire.
    # --------------------------------------------------------

    join_transition = get_or_create_transition(
        spn,
        (
            f"{branch_prefix}_"
            f"AND_JOIN"
        ),
        id,
        has_conflict,
        conflicts,
        color=color
    )

    for completion_place in completion_places:

        _add_arc_place_to_transition(
            spn,
            completion_place,
            join_transition,
            arc_color
        )

    # --------------------------------------------------------
    # The completed AND condition becomes one alternative
    # input to the common conditional XOR join.
    # --------------------------------------------------------

    branch_completion = get_or_create_place(
        spn,
        (
            f"{branch_prefix}_"
            f"COMPLETE"
        )
    )

    _apply_place_metadata(
        branch_completion,
        id,
        has_conflict,
        conflicts,
        color
    )

    _add_arc_transition_to_place(
        spn,
        join_transition,
        branch_completion,
        arc_color
    )

    _add_arc_place_to_transition(
        spn,
        branch_completion,
        state["merge_transition"],
        arc_color
    )

    return state


# ============================================================
# OR conditional branch
#
# Existing implementation treats OR as XOR.
#
# We preserve that behavior, but make the topology valid.
# ============================================================

def _add_or_conditional_branch(
    spn,
    source_transitions,
    then_transition,
    branch_number,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    state = _ensure_conditional_merge(
        spn,
        then_transition,
        id,
        has_conflict,
        conflicts,
        color
    )

    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    branch_prefix = (
        f"CONDITIONAL_"
        f"{then_transition.label}_"
        f"OR_BRANCH_{branch_number}"
    )

    # --------------------------------------------------------
    # OR gate
    #
    # The gate has one incoming place and multiple alternative
    # transitions. This preserves the existing OR-as-XOR
    # interpretation without creating unconnected places.
    # --------------------------------------------------------

    gate_input = get_or_create_place(
        spn,
        f"{branch_prefix}_INPUT"
    )

    _apply_place_metadata(
        gate_input,
        id,
        has_conflict,
        conflicts,
        color
    )

    gate_transition = get_or_create_transition(
        spn,
        f"{branch_prefix}_GATE",
        id,
        has_conflict,
        conflicts,
        color=color
    )

    # --------------------------------------------------------
    # We need a token source for this branch.
    #
    # Since the OR condition itself consists of the source
    # transitions, use their completion places as the actual
    # alternatives. Do not create a fake disconnected split.
    # --------------------------------------------------------

    alternative_completion_places = []

    for source in source_transitions:

        completion_place = get_or_create_place(
            spn,
            (
                f"{branch_prefix}_"
                f"{source.label}_DONE"
            )
        )

        _apply_place_metadata(
            completion_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            source,
            completion_place,
            arc_color
        )

        alternative_completion_places.append(
            completion_place
        )

    # Each alternative goes directly to the common merge.
    #
    # This is the cleanest Petri-net representation of:
    #
    # A OR B -> C
    #
    # because the common merge transition is enabled by
    # whichever alternative completes.
    for completion_place in alternative_completion_places:

        _add_arc_place_to_transition(
            spn,
            completion_place,
            state["merge_transition"],
            arc_color
        )

    return state


# ============================================================
# Ordinary XOR split
#
# Used only when the expression explicitly contains OR.
# ============================================================

def connect_xor_split(
    spn,
    source,
    targets,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    split_place = get_or_create_place(
        spn,
        (
            f"XOR_SPLIT_"
            f"{source.label}_TO_"
            f"{'_'.join(t.label for t in targets)}"
        )
    )

    _apply_place_metadata(
        split_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _connect_transition_place(
        spn,
        source,
        split_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    for target in targets:

        _connect_place_transition(
            spn,
            split_place,
            target,
            id,
            has_conflict,
            conflicts,
            color
        )

    return split_place


# ============================================================
# Ordinary AND split
#
# Kept for compatibility with the existing implementation.
#
# IMPORTANT:
# Unlike the old implementation, this creates an actual
# transition that consumes the input place and produces one
# separate place for each target.
# ============================================================

def connect_and_split(
    spn,
    source,
    targets,
    id,
    has_conflict,
    conflicts,
    color="forestgreen"
):
    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    split_transition = get_or_create_transition(
        spn,
        (
            f"AND_SPLIT_"
            f"{source.label}_TO_"
            f"{'_'.join(t.label for t in targets)}"
        ),
        id,
        has_conflict,
        conflicts,
        color=color
    )

    input_place = get_or_create_place(
        spn,
        (
            f"AND_SPLIT_"
            f"{source.label}_INPUT"
        )
    )

    _apply_place_metadata(
        input_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _add_arc_place_to_transition(
        spn,
        input_place,
        split_transition,
        arc_color
    )

    # Source -> input place.
    _add_arc_transition_to_place(
        spn,
        source,
        input_place,
        arc_color
    )

    # One distinct output place per branch.
    for target in targets:

        branch_place = get_or_create_place(
            spn,
            (
                f"AND_SPLIT_"
                f"{source.label}_"
                f"TO_{target.label}"
            )
        )

        _apply_place_metadata(
            branch_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            split_transition,
            branch_place,
            arc_color
        )

        _add_arc_place_to_transition(
            spn,
            branch_place,
            target,
            arc_color
        )

    return split_transition


# ============================================================
# AND JOIN
#
# Genuine AND join:
#
# A -> P_A --\
# B -> P_B ---+-> AND_JOIN -> P -> target
#
# ============================================================

def connect_and_join(
    spn,
    sources,
    target,
    id,
    has_conflict,
    conflicts,
    return_transition=False,
    color="forestgreen"
):
    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    source_labels = "_".join(
        str(t.label)
        for t in sources
    )

    gateway_label = (
        f"AND_JOIN_{source_labels}"
    )

    join_transition = get_or_create_transition(
        spn,
        gateway_label,
        id,
        has_conflict,
        conflicts,
        color=color
    )

    for source in sources:

        source_place = get_or_create_place(
            spn,
            (
                f"AND_JOIN_"
                f"{source.label}_"
                f"TO_{gateway_label}"
            )
        )

        _apply_place_metadata(
            source_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            source,
            source_place,
            arc_color
        )

        _add_arc_place_to_transition(
            spn,
            source_place,
            join_transition,
            arc_color
        )

    if target is None:
        return join_transition

    output_place = get_or_create_place(
        spn,
        (
            f"AND_JOIN_"
            f"{source_labels}_"
            f"TO_{target.label}"
        )
    )

    _apply_place_metadata(
        output_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _add_arc_transition_to_place(
        spn,
        join_transition,
        output_place,
        arc_color
    )

    _add_arc_place_to_transition(
        spn,
        output_place,
        target,
        arc_color
    )

    return join_transition if return_transition else output_place


# ============================================================
# XOR JOIN
#
# Multiple alternative places feed ONE transition.
#
# A -> P_A --\
#              \
# B -> P_B ----> XOR_JOIN -> P -> target
#
# ============================================================

def connect_xor_join(
    spn,
    sources,
    target,
    id,
    has_conflict,
    conflicts,
    return_transition=False,
    color="forestgreen"
):
    arc_color = (
        "firebrick"
        if has_conflict
        else color
    )

    source_labels = "_".join(
        str(t.label)
        for t in sources
    )

    join_transition = get_or_create_transition(
        spn,
        f"XOR_JOIN_{source_labels}",
        id,
        has_conflict,
        conflicts,
        color=color
    )

    for source in sources:

        source_place = get_or_create_place(
            spn,
            (
                f"XOR_JOIN_"
                f"{source.label}_"
                f"TO_{join_transition.label}"
            )
        )

        _apply_place_metadata(
            source_place,
            id,
            has_conflict,
            conflicts,
            color
        )

        _add_arc_transition_to_place(
            spn,
            source,
            source_place,
            arc_color
        )

        _add_arc_place_to_transition(
            spn,
            source_place,
            join_transition,
            arc_color
        )

    if target is None:
        return join_transition

    output_place = get_or_create_place(
        spn,
        (
            f"XOR_JOIN_"
            f"{source_labels}_"
            f"TO_{target.label}"
        )
    )

    _apply_place_metadata(
        output_place,
        id,
        has_conflict,
        conflicts,
        color
    )

    _add_arc_transition_to_place(
        spn,
        join_transition,
        output_place,
        arc_color
    )

    _add_arc_place_to_transition(
        spn,
        output_place,
        target,
        arc_color
    )

    return join_transition if return_transition else output_place


# ============================================================
# Main integration
# ============================================================

def integrate_process_flow(
    spn,
    knowledge,
    id,
    has_conflict,
    conflicts
):
    """
    Integrate one IF/THEN statement.

    Rules:

        IF A THEN B
            -> ordinary A -> B

        IF A AND B THEN C
            -> A + B -> AND_JOIN -> C

        IF A OR B THEN C
            -> XOR alternatives -> C

        IF A AND B THEN C
        IF D AND E THEN C
            -> two AND branches -> common conditional
               XOR merge -> C

    Crucially:

        A statement without AND is NOT automatically
        converted into an XOR split.
    """

    left_expr, right_expr = parse_if_then_complex(
        knowledge
    )

    if left_expr is None or right_expr is None:
        return spn

    left_type, left_labels = parse_expression(
        left_expr
    )

    right_type, right_labels = parse_expression(
        right_expr
    )

    # --------------------------------------------------------
    # Reuse/create transitions.
    # --------------------------------------------------------

    left = [
        get_or_create_transition(
            spn,
            label,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )
        for label in left_labels
    ]

    #left[0].has_conflict == True

    right = [
        get_or_create_transition(
            spn,
            label,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )
        for label in right_labels
    ]

    #right.has_conflict == True

    # ========================================================
    # SINGLE -> SINGLE
    #
    # DO NOT create XOR structure.
    #
    # This is deliberately the first and simplest case.
    # ========================================================

    if (
        left_type == "SINGLE"
        and right_type == "SINGLE"
    ):
        connect_transition_to_transition(
            spn,
            left[0],
            right[0],
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # SINGLE -> AND
    #
    # A THEN B AND C
    #
    # This is an actual AND split because the RIGHT side
    # explicitly contains AND.
    # ========================================================

    if (
        left_type == "SINGLE"
        and right_type == "AND"
    ):
        connect_and_split(
            spn,
            left[0],
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # SINGLE -> OR
    #
    # A THEN B OR C
    #
    # Explicit OR only.
    # ========================================================

    if (
        left_type == "SINGLE"
        and right_type == "OR"
    ):
        connect_xor_split(
            spn,
            left[0],
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # AND -> SINGLE
    #
    # IF A AND B THEN C
    #
    # This is a genuine conditional AND branch.
    # ========================================================

    if (
        left_type == "AND"
        and right_type == "SINGLE"
    ):
        target = right[0]

        state = _get_conditional_state(
            spn,
            target
        )

        key = _branch_key(
            "AND",
            left_labels,
            target
        )

        if not _branch_already_exists(
            state,
            key
        ):
            branch_number = (
                len(state["branches"]) + 1
            )

            _add_and_conditional_branch(
                spn,
                left,
                target,
                branch_number,
                id,
                has_conflict,
                conflicts,
                color="forestgreen"
            )

            state["branches"].append(
                {
                    "key": key,
                    "number": branch_number,
                    "type": "AND",
                    "conditions": tuple(left_labels),
                }
            )

        return spn

    # ========================================================
    # OR -> SINGLE
    # ========================================================

    if (
        left_type == "OR"
        and right_type == "SINGLE"
    ):
        target = right[0]

        state = _get_conditional_state(
            spn,
            target
        )

        key = _branch_key(
            "OR",
            left_labels,
            target
        )

        if not _branch_already_exists(
            state,
            key
        ):
            branch_number = (
                len(state["branches"]) + 1
            )

            _add_or_conditional_branch(
                spn,
                left,
                target,
                branch_number,
                id,
                has_conflict,
                conflicts,
                color="forestgreen"
            )

            state["branches"].append(
                {
                    "key": key,
                    "number": branch_number,
                    "type": "OR",
                    "conditions": tuple(left_labels),
                }
            )

        return spn

    # ========================================================
    # AND -> AND
    #
    # Preserve the existing behavior:
    #
    # A AND B -> AND_JOIN -> C AND D
    # ========================================================

    if (
        left_type == "AND"
        and right_type == "AND"
    ):
        and_transition = get_or_create_transition(
            spn,
            "_AND_".join(left_labels),
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        connect_and_join(
            spn,
            left,
            and_transition,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        connect_and_split(
            spn,
            and_transition,
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # AND -> OR
    # ========================================================

    if (
        left_type == "AND"
        and right_type == "OR"
    ):
        and_transition = get_or_create_transition(
            spn,
            "_AND_".join(left_labels),
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        connect_and_join(
            spn,
            left,
            and_transition,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        connect_xor_split(
            spn,
            and_transition,
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # OR -> AND
    # ========================================================

    if (
        left_type == "OR"
        and right_type == "AND"
    ):
        xor_gateway = connect_xor_join(
            spn,
            left,
            None,
            id,
            has_conflict,
            conflicts,
            return_transition=True,
            color="forestgreen"
        )

        connect_and_split(
            spn,
            xor_gateway,
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    # ========================================================
    # OR -> OR
    # ========================================================

    if (
        left_type == "OR"
        and right_type == "OR"
    ):
        xor_gateway = connect_xor_join(
            spn,
            left,
            None,
            id,
            has_conflict,
            conflicts,
            return_transition=True,
            color="forestgreen"
        )

        connect_xor_split(
            spn,
            xor_gateway,
            right,
            id,
            has_conflict,
            conflicts,
            color="forestgreen"
        )

        return spn

    return spn