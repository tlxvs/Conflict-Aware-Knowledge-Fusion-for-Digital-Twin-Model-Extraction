import re
from pyspn.components.spn import SPN, Place, Transition
import pandas as pd
import json

from model_extraction.expert_knowledge_integration_helper import (
    transition_has_condition,
    transition_has_distribution,
    get_transition,
    get_component_transitions,
    get_transition_input_places,
    get_transition_output_places,
)


def integrate_state_deviation(spn, knowledge, id, has_conflict, conflicts):
    """
    Integrate state deviation knowledge as a guard
    on a failure transition.
    Behaviour:
    - If FAIL_component exists:
          add guard to it
    - If no reliability model exists:
          create FAIL_component,
          FAILED_component place,
          REPAIR_component transition,
          then attach guard.
    """
  
    if knowledge is None:
        return spn

    knowledge = str(knowledge).strip()

    # --------------------------------------------------
    # Extract IF condition
    # --------------------------------------------------

    match = re.search(r"IF\s+(.*?)\s+THEN", knowledge, flags=re.IGNORECASE | re.DOTALL)

    if not match:

        print(f"Cannot extract condition from: {knowledge}")

        return spn

    condition = match.group(1).strip()

    # --------------------------------------------------
    # Extract component
    # --------------------------------------------------

    match = re.search(r"\bIN\b\s+(.+)$", knowledge, flags=re.IGNORECASE)

    if not match:

        print(f"Cannot extract component from: {knowledge}")

        return spn

    component = match.group(1).strip()

    component = component.rstrip(".,; ")

    # --------------------------------------------------
    # Find existing failure transition
    # --------------------------------------------------

    failure_transition = None

    failure_label = f"FAIL_{component}"

    for transition in spn.transitions:

        if transition.label.lower() == failure_label.lower():

            failure_transition = transition
            break

    # --------------------------------------------------
    # Reliability model missing
    # --------------------------------------------------

    if failure_transition is None:

        print(
            f"No reliability model found for {component}. "
            f"Creating failure/repair structure."
        )

        create_failure_structure(spn, component, id, has_conflict, conflicts)

        # search again

        for transition in spn.transitions:

            if transition.label.lower() == failure_label.lower():

                failure_transition = transition
                break

    # --------------------------------------------------
    # Fallback matching
    # --------------------------------------------------

    if failure_transition is None:

        for transition in spn.transitions:

            if not transition.label.startswith("FAIL_"):
                continue

            if component.lower() in transition.label.lower():

                failure_transition = transition
                failure_transition.has_conflict = has_conflict
                break

    if failure_transition is None:

        print(f"Could not create/find failure transition " f"for {component}")

        return spn

    # --------------------------------------------------
    # Add guard condition
    # --------------------------------------------------

    if transition_has_condition(failure_transition):

        #
        # Existing guarded transition:
        # clone it and attach another condition.
        #

        print("Existing condition found. " "Creating additional failure transition.")

        failure_transition = clone_failure_transition(
            spn, failure_transition, condition, id, has_conflict, conflicts
        )
        failure_transition.has_conflict = has_conflict

    elif transition_has_distribution(failure_transition):

        #
        # Existing reliability transition:
        #

        print("Existing reliability transition found. " "Creating guarded clone.")

        failure_transition = clone_failure_transition(
            spn, failure_transition, condition, id, has_conflict, conflicts
        )

        failure_transition.has_conflict = has_conflict

    else:

        failure_transition.condition_label = "*" + condition
        failure_transition.t_type = "I"
        failure_transition.has_conflict = has_conflict

    if has_conflict == True:
        failure_transition.color = "firebrick"
    else:
        failure_transition.color = "darkcyan"

    failure_transition.has_conflict = has_conflict

    if has_conflict == True:
        failure_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        failure_transition.conflicted_with = conflicts
    failure_transition.source = id
    

    if hasattr(spn, "transitions"):

        for transition in spn.transitions:

            if transition.label in [f"FAIL_{component}", f"REPAIR_{component}"]:

                if has_conflict == True:
                    transition.color = "firebrick"
                else:
                    transition.color = "darkcyan"

                transition.has_conflict = has_conflict

                if has_conflict == True:
                    transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
                else: 
                    transition.conflicted_with = conflicts
               
    print(
        f"Integrated state deviation: "
        f"IF {condition} "
        f"-> {failure_transition.label}"
    )

    return spn


def create_failure_structure(
    spn,
    component,
    id,
    has_conflict,
    conflicts,
    failure_distribution=None,
    repair_distribution=None,
):
    """
    Creates reliability loop model
    """

    fail_label = f"FAIL_{component}"

    repair_label = f"REPAIR_{component}"

    failed_place_label = f"FAILED_{component}"

    available_place_label = f"AVAILABLE_{component}"

    if get_transition(spn, fail_label):
        return

    #
    # Timed failure transition
    #

    fail_transition = Transition(
        fail_label,
        "I",
    )
    if has_conflict == True:
        fail_transition.color = "firebrick"
    else:
        fail_transition.color = "darkcyan"

    fail_transition.has_conflict = has_conflict

    if has_conflict == True:
        fail_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        fail_transition.conflicted_with = conflicts

    fail_transition.source = id

    fail_transition.distribution = failure_distribution

    #
    # Timed repair transition
    #

    repair_transition = Transition(
        repair_label,
        "T",
    )
    if has_conflict == True:
        repair_transition.color = "firebrick"
    else:
        repair_transition.color = "darkcyan"
    repair_transition.has_conflict = has_conflict

    if has_conflict == True:
        repair_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        repair_transition.conflicted_with = conflicts
    repair_transition.source = id

    repair_transition.distribution = repair_distribution

    #
    # Places
    #

    failed_place = Place(
        failed_place_label,
        0,
    )
    if has_conflict == True:
        failed_place.color = "firebrick"
    else:
        failed_place.color = "darkcyan"
        
    failed_place.has_conflict = has_conflict

    if has_conflict == True:
        failed_place.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        failed_place.conflicted_with = conflicts
    failed_place.source = id

    available_place = Place(
        available_place_label,
        1,
    )
    if has_conflict == True:
        available_place.color = "firebrick"
    else:
        available_place.color = "darkcyan"

    available_place.has_conflict = has_conflict

    if has_conflict == True:
        available_place.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        available_place.conflicted_with = conflicts

    available_place.source = id

    spn.add_transition(fail_transition)

    spn.add_transition(repair_transition)

    spn.add_place(failed_place)

    spn.add_place(available_place)

    #
    # AVAILABLE -> FAIL
    #

    spn.add_input_arc(available_place, fail_transition, color="darkcyan")

    #
    # FAIL -> FAILED
    #

    spn.add_output_arc(fail_transition, failed_place, color="darkcyan")

    #
    # FAILED -> REPAIR
    #

    spn.add_input_arc(failed_place, repair_transition, color="darkcyan")

    #
    # REPAIR -> AVAILABLE
    #

    spn.add_output_arc(repair_transition, available_place, color="darkcyan")

    for transition in get_component_transitions(spn, component):

        if "choice" not in transition.label and "loop" not in transition.label:

         spn.add_inhibitor_arc(transition, failed_place, color="darkcyan")

    if not hasattr(spn, "failed_components"):

        spn.failed_components = set()

    spn.failed_components.add(component)


def clone_failure_transition(spn, original_transition, condition, id, has_conflict, conflicts):
    """
    Clone a failure transition and reuse
    the same input/output places.
    """

    base_label = original_transition.label

    counter = 2

    new_label = f"{base_label}_{counter}"

    while spn.get_transition_by_label(new_label):

        counter += 1
        new_label = f"{base_label}_{counter}"

    new_transition = Transition(new_label, "I")

    new_transition.condition_label = "*" + condition
    new_transition.has_conflict = has_conflict

    new_transition.t_type = "I"
    if has_conflict == True:
        new_transition.color = "firebrick"
    else:
        new_transition.color = "darkcyan"

    new_transition.has_conflict = has_conflict

    if has_conflict == True:
        new_transition.conflicted_with = "[!] -> " + str(id) + " vs. " + ", ".join(str(c["statement_id"]) for c in conflicts)
    else: 
        new_transition.conflicted_with = conflicts

    new_transition.source = id

    if hasattr(original_transition, "distribution"):

        new_transition.distribution = original_transition.distribution
    #
    # Add transition to SPN
    #

    spn.add_transition(new_transition)

    #
    # Copy input places
    #

    for place in get_transition_input_places(spn, original_transition):

        spn.add_input_arc(place, new_transition, color="darkcyan")

    #
    # Copy output places
    #

    for place in get_transition_output_places(spn, original_transition):

        spn.add_output_arc(new_transition, place, color="darkcyan")

    print(
        "Created cloned failure transition:",
        new_transition.label,
        new_transition.condition_label,
    )

    return new_transition
