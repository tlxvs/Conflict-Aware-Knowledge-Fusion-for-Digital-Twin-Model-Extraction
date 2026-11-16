import re
from pyspn.components.spn import SPN, Place, Transition
import pandas as pd
import json


# ============================================================
# Metadata
# ============================================================

def _set_transition_metadata(
    transition,
    id,
    has_conflict,
    conflicts
):
    """
    Apply the same metadata that the original implementation used.
    """

    if has_conflict:
        transition.color = "firebrick"
    else:
        transition.color = "darkorange"

    transition.has_conflict = has_conflict

    if has_conflict:
        transition.conflicted_with = (
            "[!] - "
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
# Find transitions belonging to the same logical event
# ============================================================

def _find_event_transitions(spn, event):
    """
    Find all transitions belonging to the same logical event.

    Example:

        EKS1
        EKS1_2
        EKS1_3

    are all alternatives of EKS1.

    Importantly, EKS10 is NOT considered an alternative of EKS1.
    """

    event_lower = str(event).strip().lower()

    transitions = []

    for transition in spn.transitions:

        label = str(
            transition.label
        ).strip()

        label_lower = label.lower()

        # Original transition
        if label_lower == event_lower:
            transitions.append(transition)
            continue

        # Clones: EKS1_2, EKS1_3, ...
        match = re.fullmatch(
            re.escape(event_lower) + r"_(\d+)",
            label_lower
        )

        if match:
            transitions.append(transition)

    # Keep original transition first
    transitions.sort(
        key=lambda t: (
            0
            if t.label.lower() == event_lower
            else int(
                t.label.rsplit("_", 1)[1]
            )
        )
    )

    return transitions


# ============================================================
# Arc helpers
# ============================================================

def _arc_target(arc):
    """
    Get the target node from the actual PySPN arc API.

    InputArc:
        from_place -> to_transition

    OutputArc:
        from_transition -> to_place

    NOTE: some PySPN arc classes define both `to_place` and
    `to_transition` on a shared base (with the unused one set
    to None), so we must check the *value*, not just whether
    the attribute exists.
    """

    to_place = getattr(arc, "to_place", None)
    if to_place is not None:
        return to_place

    to_transition = getattr(arc, "to_transition", None)
    if to_transition is not None:
        return to_transition

    return None

def _arc_multiplicity(arc):
    """
    Preserve the multiplicity of an existing arc.
    """

    return getattr(
        arc,
        "multiplicity",
        1
    )


# ============================================================
# Get existing arcs
# ============================================================

def _get_input_arcs(spn, transition):
    return list(transition.input_arcs)

# ============================================================
# Find objects
# ============================================================

# ============================================================
# Arc existence
# ============================================================

def _input_arc_exists(spn, place, transition):
    return any(
        arc.from_place is place
        and arc.to_transition is transition
        for arc in transition.input_arcs
    )


def _output_arc_exists(spn, transition, place):
    return any(
        arc.from_transition is transition
        and arc.to_place is place
        for arc in transition.output_arcs
    )

# ============================================================
# Arc creation
# ============================================================

def _add_input_arc(
    spn,
    place,
    transition,
    multiplicity=1
):
    """
    Add:

        Place -> Transition

    ONLY if that exact connection does not already exist.

    Existing arcs:
        - are never removed
        - are never modified
        - are never duplicated
    """

    if place is None:
        return

    if transition is None:
        return

    # --------------------------------------------------------
    # Existing arc?
    # --------------------------------------------------------

    if _input_arc_exists(
        spn,
        place,
        transition
    ):

        return

    # --------------------------------------------------------
    # Create new arc
    # --------------------------------------------------------

    if multiplicity != 1:

        spn.add_input_arc(
            place,
            transition,
            multiplicity=multiplicity
        )

    else:

        spn.add_input_arc(
            place,
            transition
        )

def _add_output_arc(
    spn,
    transition,
    place,
    multiplicity=1
):
    """
    Add:

        Transition -> Place

    ONLY if that exact connection does not already exist.

    Existing arcs:
        - are never removed
        - are never modified
        - are never duplicated

    New arcs inherit the transition's color/conflict metadata.
    """

    if transition is None:
        return

    if place is None:
        return

    # --------------------------------------------------------
    # Existing arc?
    # --------------------------------------------------------

    if _output_arc_exists(
        spn,
        transition,
        place
    ):
        return

    # --------------------------------------------------------
    # Metadata for the NEW arc
    # --------------------------------------------------------

    color = getattr(
        transition,
        "color",
        "black"
    )

    


    has_conflict = getattr(
        transition,
        "has_conflict",
        False
    )

    if color == "firebrick":
        has_conflict == True

    # --------------------------------------------------------
    # Create NEW arc
    # --------------------------------------------------------

    spn.add_output_arc(
        transition,
        place,
        multiplicity=multiplicity,
        color=color,
        has_conflict=has_conflict
    )

# ============================================================
# Clone transition
# ============================================================

def _clone_transition(
    spn,
    original_transition,
    event
):
    """
    Clone the transition and preserve its existing output arcs.

    Existing functionality is preserved:
        - Original transition is untouched.
        - Original arcs are untouched.
        - Clone receives the same output places as the original.
        - Arc multiplicity is preserved.
        - Duplicate arcs are not created.
        - Transition metadata is copied.
    """

    # --------------------------------------------------------
    # Determine next clone number
    # --------------------------------------------------------

    clone_number = 2

    while any(
        str(t.label).lower()
        == f"{event}_{clone_number}".lower()
        for t in spn.transitions
    ):
        clone_number += 1

    clone_label = f"{event}_{clone_number}"

    print(
        f"Cloning transition "
        f"{original_transition.label} -> {clone_label}"
    )

    # --------------------------------------------------------
    # Create clone
    # --------------------------------------------------------

    clone = Transition(
        clone_label,
        original_transition.t_type
    )

    # --------------------------------------------------------
    # Copy transition attributes
    # --------------------------------------------------------

    for attr in (
        "color",
        "fontcolor",
        "has_conflict",
        "conflicted_with",
        "source",
        "weight",
        "probability",
        "distribution",
    ):

        if hasattr(
            original_transition,
            attr
        ):
            setattr(
                clone,
                attr,
                getattr(
                    original_transition,
                    attr
                )
            )

    # --------------------------------------------------------
    # Add clone to SPN
    # --------------------------------------------------------

    spn.add_transition(
        clone
    )

    #spn.add_output_arc(#
    #
    #)

    # --------------------------------------------------------
    # IMPORTANT:
    # Copy the ORIGINAL output arcs to the clone.
    #
    # Original:
    #
    #     original_transition -> output_place
    #
    # Clone:
    #
    #     clone -> output_place
    #
    # Existing original arcs are NOT touched.
    # --------------------------------------------------------

    for arc in list(
        original_transition.output_arcs
    ):

        output_place = _arc_target(
            arc
        )

        multiplicity = _arc_multiplicity(
            arc
        )

        if output_place is None:
            continue

        _add_output_arc(
            spn,
            clone,
            output_place,
            multiplicity
        )

        print(
            f"Cloned output arc: "
            f"{clone.label} -> "
            f"{output_place.label}"
        )

    print(
        f"Successfully created clone "
        f"{clone.label}"
    )

    return clone


# ============================================================
# Create XOR split/join
# ============================================================


def _create_xor_structure(
    spn,
    original_transition,
    clone_transition
):
    """
    Create the XOR alternatives using the existing places.

    XOR split:

        P_before
          | \
          |  \
          v   v
         T1  T2

    XOR join:

         T1  T2
          \  /
           \/
        P_after

    Existing arcs are NEVER removed.
    Existing arcs are NEVER modified.
    Duplicate arcs are NEVER created.
    """

    print(
        f"Creating XOR structure for "
        f"{original_transition.label} "
        f"and {clone_transition.label}"
    )

    # --------------------------------------------------------
    # Existing input arcs of original transition
    # --------------------------------------------------------

    input_arcs = list(
        original_transition.input_arcs
    )

    if not input_arcs:

        raise RuntimeError(
            f"No input place found for "
            f"{original_transition.label}"
        )

    # --------------------------------------------------------
    # Existing output arcs of original transition
    # --------------------------------------------------------

    output_arcs = list(
        original_transition.output_arcs
    )

    if not output_arcs:

        raise RuntimeError(
            f"No output place found for "
            f"{original_transition.label}"
        )

    # --------------------------------------------------------
    # Find/create XOR split transition
    #
    # Keep your existing XOR transition functionality.
    # --------------------------------------------------------

    xor_split_label = (
        f"{original_transition.label}_XOR_SPLIT"
    )

    xor_split = spn.get_transition_by_label(
        xor_split_label
    )


    # --------------------------------------------------------
    # Find/create XOR join transition
    # --------------------------------------------------------

    xor_join_label = (
        f"{original_transition.label}_XOR_JOIN"
    )

    xor_join = spn.get_transition_by_label(
        xor_join_label
    )

    
    # --------------------------------------------------------
    # XOR SPLIT
    #
    # The existing input place is the XOR choice place.
    #
    # We keep:
    #
    #     previous_place -> original_transition
    #
    # and add:
    #
    #     previous_place -> clone_transition
    #
    # No branch place is created.
    # --------------------------------------------------------

    for arc in input_arcs:

        previous_place = arc.from_place

        if clone_transition.has_conflict == True:

            previous_place.color = "firebrick"
            previous_place.has_conflict = True

        multiplicity = getattr(
            arc,
            "multiplicity",
            1
        )

        _add_input_arc(
            spn,
            previous_place,
            clone_transition,
            multiplicity
        )

        print(
            f"XOR split: "
            f"{previous_place.label} -> "
            f"{original_transition.label}"
        )

        print(
            f"XOR split: "
            f"{previous_place.label} -> "
            f"{clone_transition.label}"
        )

    # --------------------------------------------------------
    # XOR JOIN
    #
    # The existing output place is the common destination.
    #
    # We keep:
    #
    #     original_transition -> output_place
    #
    # and add:
    #
    #     clone_transition -> output_place
    #
    # No join place is created.
    # --------------------------------------------------------

    for arc in output_arcs:

        output_place = arc.to_place

        multiplicity = getattr(
            arc,
            "multiplicity",
            1
        )

        _add_output_arc(
            spn,
            clone_transition,
            output_place,
            multiplicity
        )

        print(
            f"XOR join: "
            f"{original_transition.label} -> "
            f"{output_place.label}"
        )

        print(
            f"XOR join: "
            f"{clone_transition.label} -> "
            f"{output_place.label}"
        )

    print(
        f"XOR structure created for "
        f"{original_transition.label}"
    )

    return xor_split, xor_join

def _add_clone_to_existing_xor(
    spn,
    event,
    clone_transition
):
    """
    Add another XOR alternative.

    All alternatives consume from the same previous place.

        previous_place
          |     |     |
          v     v     v
         T1    T2    T3
          |     |     |
         J1    J2    J3
          |     |     |
          +-----+-----+
                |
             XOR_JOIN

    Existing arcs are never removed.
    Duplicate arcs are never created.
    """

    original_transition = spn.get_transition_by_label(
        event
    )

    if original_transition is None:
        raise RuntimeError(
            f"Cannot add clone {clone_transition.label}: "
            f"original transition {event} does not exist."
        )

    xor_join = spn.get_transition_by_label(
        f"{event}_XOR_JOIN"
    )

    if xor_join is None:
        raise RuntimeError(
            f"Cannot add clone {clone_transition.label}: "
            f"{event}_XOR_JOIN does not exist."
        )

    # --------------------------------------------------------
    # Find the SAME previous place used by the original.
    # --------------------------------------------------------

    input_arcs = list(
        original_transition.input_arcs
    )

    if not input_arcs:
        raise RuntimeError(
            f"No input place found for "
            f"{original_transition.label}"
        )

    # --------------------------------------------------------
    # Connect clone to the same previous place.
    #
    # This creates the actual XOR alternative.
    # --------------------------------------------------------

    for arc in input_arcs:

        previous_place = arc.from_place

        _add_input_arc(
            spn,
            previous_place,
            clone_transition,
            arc.multiplicity
        )

    # --------------------------------------------------------
    # Determine next join place.
    # --------------------------------------------------------

    branch_number = 1

    while spn.get_place_by_label(
        f"{event}_XOR_JOIN_{branch_number}"
    ) is not None:

        branch_number += 1

    join_place = Place(
        f"{event}_XOR_JOIN_{branch_number}",
        0
    )

    join_place.color = "firebrick"
    
    join_place.has_conflict = True

    spn.add_place(
        join_place
    )

    # --------------------------------------------------------
    # Clone -> join place
    # --------------------------------------------------------

    _add_output_arc(
        spn,
        clone_transition,
        join_place,
        color = "firebrick",
        has_conflict = True

    )

    # --------------------------------------------------------
    # Join place -> XOR JOIN
    # --------------------------------------------------------

    _add_input_arc(
        spn,
        join_place,
        xor_join
    )

    print(
        f"Added TRUE XOR alternative "
        f"{clone_transition.label} "
        f"to {original_transition.label}"
    )

# ============================================================
# Connect previous place to XOR split
# ============================================================

# ============================================================
# Add clone to existing XOR
# ============================================================

# ============================================================
# MAIN FUNCTION
# ============================================================

def integrate_temporal_and_probabilistic(
    spn,
    knowledge,
    id,
    has_conflict,
    conflicts
):
    """
    Integrate temporal and probabilistic knowledge.

    Existing functionality:
    ------------------------

    - Existing transition is updated.
    - Missing transition is created.
    - Distribution creates timed transition (T).
    - Weight/probability creates immediate transition (I).
    - Conflict metadata is preserved.
    - Source information is preserved.
    - Conflict information is preserved.

    Additional functionality:
    --------------------------

    - If the same IF event occurs again, the transition is cloned.
    - Each clone receives its own distribution/weight.
    - The original transition and clone are connected through
      an XOR split and XOR join.
    - Existing arcs are never removed.
    - Existing arcs are never modified.
    - Duplicate arcs are not created.
    """


    if knowledge is None:
        return spn

    knowledge = str(
        knowledge
    ).strip()

    if not knowledge:
        return spn

    # ========================================================
    # Extract IF event
    # ========================================================

    match = re.search(
        r"IF\s+(.*?)\s+THEN",
        knowledge,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:

        print(
            f"Cannot extract temporal event: "
            f"{knowledge}"
        )

        return spn

    event = match.group(1).strip()

    # ========================================================
    # Extract THEN information
    # ========================================================

    match = re.search(
        r"THEN\s+(.*)$",
        knowledge,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:

        print(
            f"Cannot extract temporal information: "
            f"{knowledge}"
        )

        return spn

    temporal_information = (
        match.group(1).strip()
    )

    # ========================================================
    # Find existing transitions
    # ========================================================

    existing_transitions = (
        _find_event_transitions(
            spn,
            event
        )
    )

    # ========================================================
    # First occurrence
    # ========================================================

    if not existing_transitions:

        print(
            f"No transition found for "
            f"{event}. Creating."
        )

        transition = Transition(
            event,
            "I"
        )

        _set_transition_metadata(
            transition,
            id,
            has_conflict,
            conflicts
        )

        spn.add_transition(
            transition
        )

        is_clone = False

    # ========================================================
    # Second/subsequent occurrence
    # ========================================================

    else:

        original_transition = (
            existing_transitions[0]
        )

        print(
            f"Transition {event} already exists."
        )

        print(
            f"Existing alternatives: "
            f"{[t.label for t in existing_transitions]}"
        )

        # ----------------------------------------------------
        # Clone original transition
        # ----------------------------------------------------

        transition = _clone_transition(
            spn,
            original_transition,
            event
        )

        # ----------------------------------------------------
        # Apply NEW statement metadata
        # ----------------------------------------------------

        _set_transition_metadata(
            transition,
            id,
            has_conflict,
            conflicts
        )

        is_clone = True

    # ========================================================
    # Distribution
    # ========================================================

    distribution_match = re.search(
        r"distribution\s*=\s*([a-zA-Z_]+)",
        temporal_information,
        flags=re.IGNORECASE
    )

    if distribution_match:

        distribution_name = (
            distribution_match
            .group(1)
            .strip()
        )

        parameters = {}

        parameter_matches = re.findall(
            r"([a-zA-Z_]+)\s*=\s*([^,]+)",
            temporal_information
        )

        for key, value in parameter_matches:

            if key.lower() != "distribution":

                parameters[
                    key.strip()
                ] = value.strip()

        # ----------------------------------------------------
        # Timed transition
        # ----------------------------------------------------

        transition.t_type = "T"

        transition.distribution = {
            distribution_name: parameters
        }

        _set_transition_metadata(
            transition,
            id,
            has_conflict,
            conflicts
        )

        print(
            f"Added distribution "
            f"{distribution_name} "
            f"to {transition.label}"
        )

    # ========================================================
    # Weight / probability
    # ========================================================

    else:

        weight_match = re.search(
            r"(weight|probability)\s*=\s*([0-9.]+)\s*%?",
            temporal_information,
            flags=re.IGNORECASE
        )

        if weight_match:

            value_type = (
                weight_match
                .group(1)
                .lower()
            )

            weight = float(
                weight_match.group(2)
            )

            # ------------------------------------------------
            # Convert percentage probability
            # ------------------------------------------------

            if (
                value_type == "probability"
                and "%" in temporal_information
            ):

                weight = (
                    weight / 100.0
                )

            # ------------------------------------------------
            # Immediate transition
            # ------------------------------------------------

            transition.t_type = "I"

            transition.weight = (
                "w = "
                + str(weight)
            )

            transition.probability = (
                weight
            )

            _set_transition_metadata(
                transition,
                id,
                has_conflict,
                conflicts
            )

            print(
                f"Added probability/weight "
                f"{weight} "
                f"to {transition.label}"
            )

        else:

            print(
                f"Unknown temporal/probabilistic format: "
                f"{temporal_information}"
            )

    # ========================================================
    # Clone -> XOR
    # ========================================================

    if is_clone:

        already_has_xor = any(
            str(t.label)
            == f"{event}_XOR_SPLIT"
            for t in spn.transitions
        )

        if not already_has_xor:

            _create_xor_structure(
                spn,
                existing_transitions[0],
                transition
            )

        else:

            _add_clone_to_existing_xor(
                spn,
                event,
                transition
            )

    return spn