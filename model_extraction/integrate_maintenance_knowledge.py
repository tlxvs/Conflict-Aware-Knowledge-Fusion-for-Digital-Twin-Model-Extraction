import re
from pyspn.components.spn import SPN, Place, Transition
import pandas as pd
import json

from model_extraction.expert_knowledge_integration_helper import (
    get_transition,
    get_component_transitions,
    get_transition_input_places,
    get_transition_output_places,
    transition_has_distribution,
    transition_has_condition,
)


def integrate_maintenance(spn, knowledge, id, has_conflict, conflicts):
    """
    Integrate maintenance knowledge as a guarded maintenance transition.
    Behaviour:
    - If MAINTENANCE_component exists:
          add guard to it
    - If no maintenance model exists:
          create it 
    - Multiple maintenance rules are supported by cloning transitions.
    """


    if knowledge is None:
        return spn

    knowledge = str(knowledge).strip()

    #
    # Extract IF condition
    #

    match = re.search(r"IF\s+(.*?)\s+THEN", knowledge, flags=re.IGNORECASE | re.DOTALL)

    if not match:

        print(f"Cannot extract maintenance condition from: {knowledge}")

        return spn

    condition = match.group(1).strip()

    #
    # Extract action
    #

    match_action = re.search(
        r"THEN\s+(.*?)\s+IN\b", knowledge, flags=re.IGNORECASE | re.DOTALL
    )

    if not match_action:

        print(f"Cannot extract maintenance action from: {knowledge}")

        return spn

    action = match_action.group(1).strip()

    #
    # Extract component
    #

    match_component = re.search(r"\bIN\b\s+(.+)$", knowledge, flags=re.IGNORECASE)

    if not match_component:

        print(f"Cannot extract maintenance component from: {knowledge}")

        return spn

    component = match_component.group(1).strip().rstrip(".,; ")

    #
    # Find maintenance transition
    #

    maintenance_label = f"MAINTAIN_{component}"

    maintenance_transition = None

    for transition in spn.transitions:

        if transition.label.lower() == maintenance_label.lower():

            maintenance_transition = transition
            break

    #
    # Create maintenance structure if missing
    #

    if maintenance_transition is None:

        print(
            f"No maintenance model found for {component}. "
            f"Creating maintenance structure."
        )

        create_maintenance_structure(spn, component, id, has_conflict, conflicts)

        for transition in spn.transitions:

            if transition.label.lower() == maintenance_label.lower():

                maintenance_transition = transition
                break

    #
    # Fallback component matching
    #

    if maintenance_transition is None:

        for transition in spn.transitions:

            if not transition.label.startswith("MAINTAIN_"):

                continue

            if component.lower() in transition.label.lower():

                maintenance_transition = transition
                break

    if maintenance_transition is None:

        print(f"Could not create/find maintenance transition " f"for {component}")

        return spn

    #
    # Attach maintenance condition
    #

    if transition_has_condition(maintenance_transition):

        print(
            "Existing maintenance condition found. "
            "Creating additional maintenance transition."
        )

        maintenance_transition = clone_maintenance_transition(
            spn, maintenance_transition, condition, has_conflict
        )

    elif transition_has_distribution(maintenance_transition):

        print(
            "Existing maintenance distribution found. "
            "Creating guarded maintenance clone."
        )

        maintenance_transition = clone_maintenance_transition(
            spn, maintenance_transition, condition, has_conflict
        )

    else:

        maintenance_transition.condition_label = "*" + condition

        maintenance_transition.t_type = "I"

    #
    # Metadata
    #
    maintenance_transition.condition = condition

    maintenance_transition.action = action

    for transition in spn.transitions:

        if transition.label.startswith("MAINTAIN_"):

            if has_conflict == True:
                transition.color = "firebrick"
            else:
                transition.color = "purple"
            transition.source = id
            transition.has_conflict = has_conflict
            if has_conflict == True:
                transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
            else: 
                transition.conflicted_with = conflicts

    for place in spn.places:

        if place.label.startswith("MAINTAINED_"):

            if has_conflict == True:
                place.color = "firebrick"
            else:
                place.color = "purple"
            place.source = id
            place.has_conflict = has_conflict
            if has_conflict == True:
                place.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
            else: 
                place.conflicted_with = conflicts

    print(
        f"Integrated maintenance: "
        f"IF {condition} "
        f"THEN {action} "
        f"IN {component}"
    )

    return spn


def create_maintenance_structure(spn, component, id, has_conflict, conflicts, maintenance_distribution=None):
    """
    Creates maintenance-related components 
    """

    maintenance_label = f"MAINTAIN_{component}"

    maintained_place_label = f"MAINTAINED_{component}"

    available_place_label = f"AVAILABLE_{component}"

    #
    # Avoid duplicate maintenance structures
    #

    if get_transition(spn, maintenance_label):

        return

    #
    # Maintenance transition
    #

    maintenance_transition = Transition(maintenance_label, "I")

    if has_conflict == True:
        maintenance_transition.color = "firebrick"
    else:
        maintenance_transition.color = "purple"
    maintenance_transition.has_conflict = has_conflict
    if has_conflict == True:
        maintenance_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        maintenance_transition.conflicted_with = conflicts

    maintenance_transition.source = id

    maintenance_transition.distribution = maintenance_distribution

    #
    # Places
    #

    maintained_place = Place(maintained_place_label, 0)

    if has_conflict == True:
        maintained_place.color = "firebrick"
    else:
        maintained_place.color = "purple"

    maintained_place.has_conflict = has_conflict

    if has_conflict == True:
        maintained_place.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        maintained_place.conflicted_with = conflicts
    

    maintained_place.source = id

    available_place = Place(available_place_label, 1)
    if has_conflict == True:
        available_place.color = "firebrick"
    else:
        available_place.color = "purple"

    available_place.has_conflict = has_conflict

    if has_conflict == True:
        available_place.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        available_place.conflicted_with = conflicts

    available_place.source = id

    #
    # Add elements
    #

    spn.add_transition(maintenance_transition)

    spn.add_place(maintained_place)

    spn.add_place(available_place)

    #
    # AVAILABLE -> MAINTENANCE
    #

    spn.add_input_arc(available_place, maintenance_transition, color="purple", has_conflict=has_conflict)

    #
    # MAINTENANCE -> MAINTAINED
    #

    spn.add_output_arc(maintenance_transition, maintained_place, color="purple", has_conflict=has_conflict)

    #
    # MAINTAINED -> AVAILABLE
    #
    # This arc represents returning to service
    # after maintenance completion.
    #

    spn.add_output_arc(maintenance_transition, available_place, color="purple", has_conflict=has_conflict)

    #
    # Prevent operation during maintenance
    #

    for transition in get_component_transitions(spn, component):

        if "maintenance" not in transition.label and "MAINTAIN" not in transition.label:

            spn.add_inhibitor_arc(transition, maintained_place, color="purple")

        if "maintenance" in transition.label or "MAINTAIN" in transition.label:

            spn.add_output_arc(maintained_place, transition, color="purple")

      
    #
    # Store metadata
    #

    if not hasattr(spn, "maintained_components"):

        spn.maintained_components = set()

    spn.maintained_components.add(component)

    print(f"Created maintenance structure for {component}")


def clone_maintenance_transition(spn, original_transition, condition, has_conflict, conflicts):
    """
    Clone a maintenance transition and reuse
    the same input/output places.
    Used when multiple maintenance IF-THEN-IN
    rules exist for the same component.
    """

    #
    # Generate unique label
    #

    base_label = original_transition.label

    counter = 2

    new_label = f"{base_label}_{counter}"

    while spn.get_transition_by_label(new_label):

        counter += 1

        new_label = f"{base_label}_{counter}"

    #
    # Create cloned transition
    #

    new_transition = Transition(new_label, "I")

    #
    # Add condition
    #

    new_transition.condition_label = "*" + condition

    new_transition.t_type = "I"

    if has_conflict == True:
        new_transition.color = "firebrick"
    else:
        new_transition.color = "purple"

    new_transition.has_conflict = has_conflict

    if has_conflict == True:
        new_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        new_transition.conflicted_with = conflicts

    new_transition.source = id

    #
    # Copy distribution if available
    #

    if hasattr(original_transition, "distribution"):

        new_transition.distribution = original_transition.distribution

    #
    # Copy metadata
    #

    new_transition.condition = condition

    #
    # Add transition to SPN
    #

    spn.add_transition(new_transition)

    #
    # Copy input places
    #

    for place in get_transition_input_places(spn, original_transition):

        spn.add_input_arc(place, new_transition, color="purple")

    #
    # Copy output places
    #

    for place in get_transition_output_places(spn, original_transition):

        spn.add_output_arc(new_transition, place, color="purple")

    #
    # Store maintenance metadata
    #

    new_transition.action = getattr(original_transition, "action", None)

    print(
        "Created cloned maintenance transition:",
        new_transition.label,
        new_transition.condition_label,
    )

    return new_transition
