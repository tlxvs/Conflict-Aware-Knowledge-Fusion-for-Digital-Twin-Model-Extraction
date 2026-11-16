import html
from typing import Iterable, Optional

from graphviz import Digraph

from .spn import *  # noqa: F401,F403  (SPN, Place, Transition, arc classes)


# =============================================================================
# Conflict display configuration — EDIT HERE
# =============================================================================
# Conflicts are shown ADDITIVELY: a flagged place/transition keeps its
# existing `color` (which encodes provenance — IoT vs. expert knowledge vs.
# fused), so that information is never lost. What changes is the border
# (dashed/thicker), the arcs (dotted, CONFLICT_ARC_COLOR) and, for
# transitions, an annotation box coloured per conflict group. Pass
# conflict_color_override="red" (or any graphviz color) to draw_spn() to
# give conflicted nodes a single unmistakable color instead.

CONFLICT_PENWIDTH = "4"
CONFLICT_MARK = "⚠"  # ⚠
CONFLICT_ARC_COLOR = "firebrick"
DEFAULT_ANNOTATION_COLOR = "#E8F0FE"

# Background colors of the annotation boxes, one per conflict group.
CONFLICT_GROUP_COLORS = [
    "#FFE0E0",
    "#E0F0FF",
    "#E2F5E2",
    "#FFF0CC",
    "#EDE0FF",
    "#DFF5F5",
]

GRAPH_MAX_WIDTH_INCHES = "6.5"
GRAPH_MAX_HEIGHT_INCHES = "1000"

GRAPH_ATTR = {
    "margin": "0.2",
    "pad": "0.2",
    "nodesep": "0.4",  # vertical gap between nodes in LR
    "ranksep": "0.9",  # horizontal gap between columns
}

NODE_FONT = {"fontsize": "28", "fontname": "Arial"}

# Sizes (inches) and line widths — EDIT HERE
PLACE_SIZE = "0.9"            # place diameter (was 0.6)
TRANSITION_WIDTH = "0.30"     # transition bar width (was 0.20)
TRANSITION_HEIGHT = "0.90"    # transition bar height (was 0.60)
ARC_PENWIDTH = "3.0"          # arc line width (graphviz default 1.0)
ARROW_SIZE = "2"            # arrowhead scale (graphviz default 1.0)


# =============================================================================
# Helpers
# =============================================================================

def _with_conflict_style(base_style: str, has_conflict: bool) -> str:
    """Adds 'dashed' to a node's style when conflicted, keeping the base style."""
    if not has_conflict:
        return base_style
    parts = [p for p in base_style.split(",") if p]
    if "dashed" not in parts:
        parts.append("dashed")
    return ",".join(parts)


def _conflict_xlabel_suffix(node) -> str:
    """Returns a label suffix from an optional `conflict_note` attribute,
    or "" if the node isn't flagged."""
    note = getattr(node, "conflict_note", None)
    if not note:
        return ""
    return "\n" + CONFLICT_MARK + " " + note


def _is_hidden(hidden_labels: Iterable[str], *names) -> bool:
    """True if any of the given node names contains a hidden label."""
    return any(h in str(name) for h in hidden_labels for name in names)


def _parse_conflict_ids(conflicted_with) -> Optional[set]:
    """Parses 'X vs. A, B' into {'A', 'B'}; returns None if not parseable."""
    if not conflicted_with:
        return None
    parts = str(conflicted_with).split("vs.", 1)
    if len(parts) != 2:
        return None
    return {x.strip() for x in parts[1].split(",") if x.strip()}


def _conflict_group_colors(transitions) -> dict:
    """Merges transitions that share a conflict partner into groups
    (connected components) and assigns one color per group.
    Returns {statement source id: color}."""
    conflict_map = {}
    for transition in transitions:
        ids = _parse_conflict_ids(transition.conflicted_with)
        if ids is not None:
            ids.add(transition.source)
            conflict_map[transition.source] = ids

    groups = []
    for ids in conflict_map.values():
        group = set(ids)
        merged = True
        while merged:
            merged = False
            for existing in groups[:]:
                if group & existing:
                    group |= existing
                    groups.remove(existing)
                    merged = True
        groups.append(group)

    colors = {}
    for i, group in enumerate(groups):
        color = CONFLICT_GROUP_COLORS[i % len(CONFLICT_GROUP_COLORS)]
        for statement_id in group:
            colors[statement_id] = color
    return colors


def _transition_extra(transition) -> Optional[str]:
    """Distribution/condition text for timed, probability for immediate."""
    if transition.t_type == "T":
        extra = format_transition_distribution(transition.distribution)
        if transition.condition_label:
            condition = str(transition.condition_label)
            extra = condition + "\n" + extra if extra else condition
        return extra

    if transition.t_type == "I":
        if getattr(transition, "weight", None) is not None:
            return ""
        if getattr(transition, "probability", None) is not None:
            return "p = " + str(transition.probability)
        return None

    print("Unknown transition type:", transition.t_type, transition.label)
    return None


def _transition_annotation(transition, extra):
    """Builds the annotation as HTML table rows (for the conflict box) and
    as plain text (for the xlabel). Category and conflict partners are only
    shown in the conflict box."""
    html_parts = [transition.label, transition.condition_label, extra]
    raw_parts = [transition.label, transition.condition_label, extra]

    source = getattr(transition, "source", None)
    if source:
        html_parts.append(source)
        raw_parts.append(source)

    html_parts += [transition.category, transition.conflicted_with]

    rows = [
        "<TR><TD>" + html.escape(str(part)) + "</TD></TR>"
        for part in html_parts if part
    ]
    raw = "\n".join(str(part) for part in raw_parts if part)
    return rows, raw


def _conflict_box(rows, background) -> str:
    return (
        "<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" "
        f"CELLPADDING=\"2\" BGCOLOR=\"{background}\">"
        + "".join(rows)
        + "</TABLE>>"
    )


def _edge(graph, tail, head, multiplicity, **attrs):
    if multiplicity > 1:
        attrs["xlabel"] = str(multiplicity)
    attrs.setdefault("penwidth", ARC_PENWIDTH)
    attrs.setdefault("arrowsize", ARROW_SIZE)
    graph.edge(tail, head, **attrs)


# =============================================================================
# Drawing
# =============================================================================

def draw_spn(
    spn: SPN,
    file="default_model",
    show=True,
    print_place_labels=False,
    rankdir="LR",
    conflict_color_override=None,
    hidden_labels: Iterable[str] = (),
    forced_conflict_labels: Iterable[str] = (),
):
    """
    Renders an SPN with graphviz to output/models/<file>.gv and .png.

    Places/transitions with `has_conflict=True` get a thicker border
    (dashed for places), dotted CONFLICT_ARC_COLOR arcs, and, for
    transitions, an annotation box whose background color identifies the
    conflict group. Objects without conflicts render normally.

    print_place_labels: kept for compatibility; places are drawn without
        labels.
    rankdir: graphviz layout direction ("LR" left-to-right, "TB" top-down).
    conflict_color_override: if set (e.g. "red"), conflicted nodes use this
        color instead of their provenance color.
    hidden_labels: nodes whose label contains any of these strings are not
        drawn, together with all arcs touching them.
    forced_conflict_labels: transitions with these labels are drawn as
        conflicted (in CONFLICT_ARC_COLOR) even if has_conflict is False.
    """
    hidden_labels = tuple(hidden_labels)
    forced_conflict_labels = set(forced_conflict_labels)

    graph = Digraph(engine="dot", graph_attr={"rankdir": rankdir, **GRAPH_ATTR})

    # --- Places --------------------------------------------------------------
    for place in spn.places:
        if _is_hidden(hidden_labels, place.label):
            continue

        has_conflict = place.has_conflict
        color = place.color
        if has_conflict and conflict_color_override:
            color = conflict_color_override

        node_kwargs = dict(
            shape="circle", label="", xlabel="",
            height=PLACE_SIZE, width=PLACE_SIZE,
            penwidth = "3.0",
            fixedsize="true", color=color, fontcolor=color, **NODE_FONT,
        )
        style = _with_conflict_style("", has_conflict)
        if style:
            node_kwargs["style"] = style
        if has_conflict:
            node_kwargs["penwidth"] = CONFLICT_PENWIDTH

        graph.node(place.label, **node_kwargs)

    # --- Transitions ---------------------------------------------------------
    group_colors = _conflict_group_colors(spn.transitions)

    for transition in spn.transitions:
        if _is_hidden(hidden_labels, transition.label):
            continue

        forced = transition.label in forced_conflict_labels
        has_conflict = transition.has_conflict or forced

        color = transition.color
        if forced:
            color = CONFLICT_ARC_COLOR
        elif has_conflict and conflict_color_override:
            color = conflict_color_override

        extra = _transition_extra(transition)
        rows, raw_annotation = _transition_annotation(transition, extra)

        node_kwargs = dict(
            shape="rectangle", label="",
            width=TRANSITION_WIDTH, height=TRANSITION_HEIGHT,
            penwidth = "3.0",
            fixedsize="true", color=color, fontcolor=color, **NODE_FONT,
        )

        # Conflicted transitions get a colored annotation box linked by a
        # dotted connector.
        if has_conflict:
            node_kwargs["penwidth"] = CONFLICT_PENWIDTH
            background = group_colors.get(str(transition.source),
                                          DEFAULT_ANNOTATION_COLOR)
            annotation_name = str(transition.label) + "__annotation"
            graph.node(annotation_name, shape="plain",
                       label=_conflict_box(rows, background), **NODE_FONT)
            graph.edge(transition.label, annotation_name, style="dotted",
                       color="dimgray", arrowhead="none", weight="100",
                       constraint="true", minlen="0")

        # Immediate transitions are filled; their xlabel is suppressed when
        # conflicted (the box shows it) or for loop/choice helpers.
        xlabel = raw_annotation
        if transition.t_type == "I":
            node_kwargs["style"] = "filled"
            if has_conflict or "loop" in xlabel or "choice" in xlabel:
                xlabel = ""

        graph.node(transition.label, xlabel=xlabel, **node_kwargs)

        # --- Arcs ------------------------------------------------------------
        for arc in transition.output_arcs:
            if _is_hidden(hidden_labels, arc.to_place.label):
                continue
            if has_conflict:
                arc_color, style = CONFLICT_ARC_COLOR, "dotted"
            else:
                arc_color, style = arc.color, "filled"
            _edge(graph, arc.from_transition.label, arc.to_place.label,
                  arc.multiplicity, color=arc_color, style=style)

        for arc in transition.input_arcs:
            if _is_hidden(hidden_labels, arc.from_place.label):
                continue
            if has_conflict or arc.color == CONFLICT_ARC_COLOR:
                arc_color, style = CONFLICT_ARC_COLOR, "dotted"
            else:
                arc_color, style = arc.color, "filled"
            _edge(graph, arc.from_place.label, arc.to_transition.label,
                  arc.multiplicity, color=arc_color, style=style)

        for arc in transition.inhibitor_arcs:
            if _is_hidden(hidden_labels, arc.from_place.label):
                continue
            _edge(graph, arc.from_place.label, arc.to_transition.label,
                  arc.multiplicity, arrowhead="dot", color="black",
                  style="solid")

    # --- Render --------------------------------------------------------------
    graph.render(f"output/models/{file}.gv", view=show)
    graph.attr(dpi="400")
    graph.render(f"output/models/{file}", view=False, format="png")
    return graph


def format_transition_distribution(distribution):
    """Formats a fitted distribution such as {"normal": (loc, scale)} as a
    multiline string for graphviz."""
    if not distribution:
        return None

    parameter_names = {
        "normal": ("loc", "scale"),
        "lognormal": ("shape", "loc", "scale"),
        "exponential": ("loc", "scale"),
    }

    lines = []
    for name, parameters in distribution.items():
        lines.append(str(name))
        if name in parameter_names:
            for label, value in zip(parameter_names[name], parameters):
                lines.append(f"{label} = {value:.4f}")
        else:
            lines.append(f"parameters = {parameters}")
    return "\n".join(lines)