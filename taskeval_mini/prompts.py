from __future__ import annotations

from textwrap import dedent


def support_action_prompt(input_text: str, context: str | None = None) -> str:
    """Return an instruction prompt for the support action extraction task.

    The model should output only JSON with specific fields.
    """
    base_instruction = """You are an assistant that reads a customer support style request
    and maps it to a structured JSON action for a ticketing system.

    Return a single JSON object with exactly these fields:
    - "intent": short string label for the main problem or request.
    - "priority": one of "low", "medium", "high".
    - "requires_human": true or false.
    - "target_system": short string for the system that should handle it, for example "billing", "network", "account", "general".
    - "sla_hours": integer number of hours for the response time target.

    Do not include any explanation. Do not wrap in markdown. Respond with JSON only.
    """
    if context:
        ctx = f"Additional account context:\n{context}\n\n"
    else:
        ctx = ""

    return dedent(
        f"""{base_instruction}

        {ctx}Customer request:
        {input_text}
        """
    ).strip()
